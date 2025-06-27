import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from critics.maac_critic import AttentionCritic
from agents.mlp_agent import MLPAgent
from agents.rnn_agent import RNNAgent
from models.model import DPGModel


def summarize_attention_critic(model: nn.Module):
    print("\n[AttentionCritic] Network Architecture Summary")
    print("=" * 63)
    print(f"Agents: {model.nagents}, Hidden Dim: {model.hidden_dim}, Attention Heads: {model.attend_heads}")
    print(f"Continuous Actions: {model.continuous}")
    print("-" * 63)

    total_params = 0

    # Per-agent components
    for i in range(model.nagents):
        print(f"\n[Agent {i + 1}]")
        agent_param_count = 0

        module_groups = [
            ("Encoding", model.critic_encoders[i]),
            ("Critic", model.critics[i]),
            ("Bias", model.biases[i]),
            ("State Encoding", model.state_encoders[i])
        ]

        for group_name, module in module_groups:
            for name, layer in module.named_children():
                if hasattr(layer, "weight"):
                    params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                    shape = list(layer.weight.shape)
                    print(f"{name}-{layer.__class__.__name__:<15} ({group_name:>12})   {shape}   {params:,}")
                    agent_param_count += params
                else:
                    print(f"{name}-{layer.__class__.__name__:<15} ({group_name:>12})   [no weights]")

        total_params += agent_param_count

    # Shared attention head extractors
    print("\n[Shared Attention Heads]")
    shared_modules = [
        ("KeyExtractor", model.key_extractors),
        ("SelectorExtractor", model.selector_extractors),
        ("ValueExtractor", model.value_extractors)
    ]

    for name, extractor_list in shared_modules:
        for idx, layer in enumerate(extractor_list):
            for sub_name, layer_sub in layer.named_children() if isinstance(layer, nn.Sequential) else [(name, layer)]:
                if hasattr(layer_sub, "weight"):
                    shape = list(layer_sub.weight.shape)
                    params = sum(p.numel() for p in layer_sub.parameters())
                    print(f"{name}-{idx + 1}-{layer_sub.__class__.__name__:<12} (Shared)   {shape}   {params:,}")
                    total_params += params

    print("=" * 63)
    print(f"Total trainable parameters: {total_params:,}")
    print("Non-trainable parameters: 0")
    print("-" * 63)
    print("Input size (MB): 0.00")
    print("Forward/backward pass size (MB): ~0.01")
    print(f"Params size (MB): {total_params * 4 / (1024 ** 2):.2f}")  # 4 bytes per float32 param
    print(f"Estimated Total Size (MB): {(total_params * 4) / (1024 ** 2) + 0.01:.2f}")
    print("=" * 63)

    # Visual flow
    print("\n[Simplified Data Flow]")
    print("1. Each agent receives its own (state, action).")
    print("2. These are encoded through the Critic Encoder => sa_encoding.")
    print("3. The state alone goes through the State Encoder => s_encoding.")
    print("4. Multi-head attention:")
    print("   - Key & Value from other agents' sa_encodings.")
    print("   - Selector from own s_encoding.")
    print("5. Attention result => attended_values.")
    print("6. Concatenate own sa_encoding + attended_values => pass to Critic.")
    print("7. s_encoding alone goes into Bias net => baseline.")
    print("8. Final output: Q(s, a) - Bias(s)")
    print("=" * 63)



def summarize_agent(agent: nn.Module, input_shape=None):
    print(f"\n[{agent.__class__.__name__}] Network Architecture Summary")
    print("=" * 63)
    
    total_params = 0

    # Print layer-by-layer summary
    for name, layer in agent.named_children():
        if hasattr(layer, 'weight'):
            shape = list(layer.weight.shape)
            params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            print(f"{name:<20} {layer.__class__.__name__:<20} {shape}   Params: {params:,}")
            total_params += params
        else:
            print(f"{name:<20} {layer.__class__.__name__:<20} [no weights]")

    print("-" * 63)
    print(f"Total trainable parameters: {total_params:,}")
    print("Non-trainable parameters: 0")
    print(f"Params size (MB): {total_params * 4 / (1024 ** 2):.2f}")
    print("=" * 63)

    # Simplified data flow
    print(f"\n[Simplified Forward Flow for {agent.__class__.__name__}]")
    







class DPGMAAC(DPGModel):
    def __init__(self, args, target_net=None):
        super(DPGMAAC, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net is not None and args.target:
            self.target_net = target_net
            self.reload_params_to_target()
        
       
            
        self.batchnorm = nn.BatchNorm1d(self.args.agent_num).to(self.device)

    def construct_model(self):
        self.construct_policy_net()
        self.construct_value_net()

    def construct_policy_net(self):
        if self.args.agent_id:
            input_shape = self.obs_dim + self.n_
        else:
            input_shape = self.obs_dim

        if self.args.agent_type == 'mlp':
            from agents.mlp_agent import MLPAgent
            Agent = MLPAgent
        elif self.args.agent_type == 'rnn':
            from agents.rnn_agent import RNNAgent
            Agent = RNNAgent
        else:
            NotImplementedError()
            
        if self.args.shared_params:
            self.policy_dicts = nn.ModuleList([ Agent(input_shape, self.args) ])
        else:
            self.policy_dicts = nn.ModuleList()
            for i in range(self.n_):
                
                input_shape = self.args.obs_dims[i]
                
                
                
                self.policy_dicts.append(Agent(input_shape, self.args))
                summarize_agent(Agent(input_shape, self.args), input_shape=input_shape)
                
                
        

    def construct_value_net(self):
        
        summarize_attention_critic( AttentionCritic(self.args).to('cpu'))
        
        print(AttentionCritic(self.args))   
        
        self.value_dicts = nn.ModuleList( [ AttentionCritic(self.args) ] )

    def get_actions(self, state, status, exploration, actions_avail, target=False, last_hid=None):
        
        target_policy = self.target_net.policy if self.args.target else self.policy
        action, hid = target_policy(state, last_hid=last_hid)
        return action, hid

       
        

    def value(self, obs, act, last_act=None, last_hid=None):
        
        
        obs_chunks = [chunk.squeeze(1) for chunk in th.chunk(obs, self.n_, dim=1)]
        act_chunks = [chunk.squeeze(1) for chunk in th.chunk(act, self.n_, dim=1)]
        sa_chunks = []
        for o, a in zip(obs_chunks, act_chunks):
           
            sa_chunks.append(th.cat((o, a), dim=-1).unsqueeze(1))
            sa_chunks.append(th.cat((o, a), dim=-1).unsqueeze(1))
            
        
        inputs = (obs_chunks, act_chunks, sa_chunks)
        
        output,weights = self.value_dicts[0](inputs)
       
        values = th.cat([o[0] for o in output], dim=1)
        return values,weights

    def get_loss(self, batch):
        (state, action, value, next_value, reward, next_state, done,
         last_step, last_hid, hid) = self.unpack_data(batch)

        action = action.view(-1, self.n_, self.act_dim)
        next_action, _ = self.get_actions(next_state, status='train', exploration=True, actions_avail=None, target=True, last_hid=last_hid)

        values,_ = self.value(state, action)
        next_values,_ = self.target_net.value(next_state, next_action, last_hid=last_hid) 
        print(next_values)
        

        
        targets = reward + self.args.gamma*next_values.detach()
        td_error = targets - values
        critic_loss = td_error.pow(2).mean()

        # Actor loss
        current_policy_action, _ = self.get_actions(state,status='train', exploration=False, actions_avail=None, target=False, last_hid=last_hid)
        policy_values,_ = self.value(state, current_policy_action)
        policy_loss = -policy_values.mean()

        return policy_loss, critic_loss
    
    

    