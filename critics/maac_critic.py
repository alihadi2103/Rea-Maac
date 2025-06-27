import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionCritic(nn.Module):
    def __init__(self, args):
        super(AttentionCritic, self).__init__()
        self.hidden_dim = args.critic_hid_size
        self.attend_heads = args.attend_heads
        assert (self.hidden_dim % self.attend_heads) == 0
        self.sa_sizes = args.sa_sizes
        self.nagents = args.agent_num
        self.continuous = args.continuous 

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.biases = nn.ModuleList()
        self.state_encoders = nn.ModuleList()

        for sdim, adim in self.sa_sizes:
            idim = sdim + adim
            odim = 1 if args.continuous else adim

            encoder = nn.Sequential()
            if args.norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim, affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, self.hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)

            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * self.hidden_dim, self.hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(self.hidden_dim, odim))
            self.critics.append(critic)

            bias = nn.Sequential()
            bias.add_module('bias_fc1', nn.Linear(self.hidden_dim, self.hidden_dim))
            bias.add_module('bias_nl', nn.LeakyReLU())
            bias.add_module('bias_fc2', nn.Linear(self.hidden_dim, 1))
            self.biases.append(bias)

            state_encoder = nn.Sequential()
            if args.norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim, self.hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = self.hidden_dim // self.attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for _ in range(self.attend_heads):
            self.key_extractors.append(nn.Linear(self.hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(self.hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(self.hidden_dim, attend_dim), nn.LeakyReLU()))

    def forward(self, inps, return_q=True, regularize=True):
        states, actions, sa = inps
        agents = range(len(self.critic_encoders))

        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, sa)]
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]

        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in agents]
        all_attend_logits = [[] for _ in agents]
        all_attend_probs = [[] for _ in agents]

        for head_idx, (curr_head_keys, curr_head_values, curr_head_selectors) in enumerate(zip(
                all_head_keys, all_head_values, all_head_selectors)):

            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys, values = [], []

                for j, (k, v) in enumerate(zip(curr_head_keys, curr_head_values)):
                    if j != a_i:
                        k = k.view(k.shape[0], -1)  # [B, D]
                        v = v.view(v.shape[0], -1)
                        keys.append(k)
                        values.append(v)

                stacked_keys = th.stack(keys)  # [N-1, B, D]
                stacked_values = th.stack(values)  # [N-1, B, D]

                stacked_keys = stacked_keys.permute(1, 2, 0)  # [B, D, N-1]
                stacked_values = stacked_values.permute(1, 2, 0)  # [B, D, N-1]
                selector = selector.view(selector.shape[0], 1, -1)  # [B, 1, D]

                attend_logits = th.matmul(selector, stacked_keys)  # [B, 1, N-1]
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)

                other_values = (stacked_values * attend_weights).sum(dim=2)  # [B, D]
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)

        all_rets = []
        for i, a_i in enumerate(agents):
            agent_rets = []

            if self.continuous:
                sa_enc_i = sa_encodings[i]
                if sa_enc_i.dim() == 3 and sa_enc_i.shape[1] == 1:
                    sa_enc_i = sa_enc_i.squeeze(1)

                cleaned_values = []
                for val in other_all_values[i]:
                    if val.dim() == 3 and val.shape[1] == 1:
                        val = val.squeeze(1)
                    cleaned_values.append(val)

                critic_in = th.cat([sa_enc_i] + cleaned_values, dim=1)
                all_q = self.critics[a_i](critic_in)
                q = all_q
            else:
                critic_in = th.cat((s_encodings[i], *other_all_values[i]), dim=1)
                all_q = self.critics[a_i](critic_in)
                int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
                q = all_q.gather(1, int_acs)

            bias_in = s_encodings[i]
            b = self.biases[a_i](bias_in)

            if return_q:
                agent_rets.append(q - b)
            if regularize:
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in all_attend_logits[i])
                agent_rets.append(attend_mag_reg.view(1, 1))

            all_rets.append(agent_rets)

        agent_att_weights = []
        for i in range(len(agents)):
            agent_weights = th.cat(all_attend_probs[i], dim=1)  # [B, H, N-1]
            agent_att_weights.append(agent_weights)

        return all_rets, agent_att_weights
