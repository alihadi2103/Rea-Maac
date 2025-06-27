import numpy as np
import torch as th
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.normal import Normal
from collections import namedtuple

import matplotlib.pyplot as plt
import seaborn as sns
import os



class GumbelSoftmax(OneHotCategorical):
    def __init__(self, logits, probs=None, temperature=0.1):
        super(GumbelSoftmax, self).__init__(logits=logits, probs=probs)
        self.eps = 1e-20
        self.temperature = temperature

    def sample_gumbel(self):
        U = self.logits.clone()
        U.uniform_(0, 1)
        return -th.log( -th.log( U + self.eps ) )

    def gumbel_softmax_sample(self):
        y = self.logits + self.sample_gumbel()
        return th.softmax( y / self.temperature, dim=-1)

    def hard_gumbel_softmax_sample(self):
        y = self.gumbel_softmax_sample()
        return (th.max(y, dim=-1, keepdim=True)[0] == y).float()

    def rsample(self):
        return self.gumbel_softmax_sample()

    def sample(self):
        return self.rsample().detach()

    def hard_sample(self):
        return self.hard_gumbel_softmax_sample()

def normal_entropy(mean, std):
    return Normal(mean, std).entropy().mean()

def multinomial_entropy(logits):
    assert logits.size(-1) > 1
    return GumbelSoftmax(logits=logits).entropy().mean()

def normal_log_density(actions, means, log_stds):
    stds = log_stds.exp()
    return Normal(means, stds).log_prob(actions)

def multinomials_log_density(actions, logits):
    assert logits.size(-1) > 1
    return GumbelSoftmax(logits=logits).log_prob(actions)

def select_action(args, logits, status='train', exploration=True, info={}):
    if args.continuous:
        act_mean = logits
        act_std = info['log_std'].exp()
        if status == 'train':
            if exploration:
                if args.action_enforcebound:
                    normal = Normal(act_mean, act_std)
                    x_t = normal.rsample()
                    y_t = th.tanh(x_t)
                    log_prob = normal.log_prob(x_t)
                    # Enforcing Action Bound
                    log_prob -= th.log(1 - y_t.pow(2) + 1e-6)
                    actions = y_t
                    return actions, log_prob
                else:
                    normal = Normal(th.zeros_like(act_mean), act_std)
                    x_t = normal.rsample()
                    log_prob = normal.log_prob(x_t)
                    # this is usually for target value
                    if info.get('clip', False):
                        actions = act_mean + th.clamp(x_t, min=-args.clip_c, max=args.clip_c)
                    else:
                        actions = act_mean + x_t
                    return actions, log_prob
            else:
                actions = act_mean
                return actions, None
        elif status == 'test':
            if args.action_enforcebound:
                x_t = act_mean
                actions = th.tanh(x_t)
                return actions, None
            else:
                actions = act_mean
                return actions, None
    else:
        if status == 'train':
            if exploration:
                if args.epsilon_softmax:
                    eps = args.softmax_eps
                    p_a = (1 - eps) * th.softmax(logits, dim=-1) + eps / logits.size(-1)
                    categorical = OneHotCategorical(logits=None, probs=p_a)
                    actions = categorical.sample()
                    log_prob = categorical.log_prob(actions)
                    return actions, log_prob
                elif args.gumbel_softmax:
                    gumbel = GumbelSoftmax(logits=logits)
                    actions = gumbel.rsample()
                    log_prob = gumbel.log_prob(actions)
                    return actions, log_prob
                else:
                    categorical = OneHotCategorical(logits=logits)
                    actions = categorical.sample()
                    log_prob = categorical.log_prob(actions)
                    return actions, log_prob
            else:
                if args.gumbel_softmax:
                    gumbel = GumbelSoftmax(logits=logits, temperature=1.0)
                    actions = gumbel.sample()
                    log_prob = gumbel.log_prob(actions)
                    return actions, log_prob
                else:
                    categorical = OneHotCategorical(logits=logits)
                    actions = categorical.sample()
                    log_prob = categorical.log_prob(actions)
                    return actions, log_prob
        elif status == 'test':
            p_a = th.softmax(logits, dim=-1)
            return  (p_a == th.max(p_a, dim=-1, keepdim=True)[0]).float(), None

def translate_action(args, action, env):
    if args.continuous:
        actions = action.detach().squeeze()
        # clip and scale action to correct range for safety
        cp_actions = th.clamp(actions, min=-1.0, max=1.0)
        low = args.action_bias - args.action_scale
        high = args.action_bias + args.action_scale
        cp_actions = 0.5 * (cp_actions + 1.0) * (high - low) + low
        cp_actions = cp_actions.cpu().numpy()
        return actions, cp_actions
    else:
        actual = [act.detach().squeeze().cpu().numpy() for act in th.unbind(action, 1)]
        return action, actual

def prep_obs(state=[]):
    state = np.array(state)
    # for single transition -> batch_size=1
    if len(state.shape) == 2:
        state = np.stack(state, axis=0)
    # for single episode
    elif len(state.shape) == 4:
        state = np.concatenate(state, axis=0)
    else:
        raise RuntimeError('The shape of the observation is incorrect.')
    return th.tensor(state).float()

def cuda_wrapper(tensor, cuda):
    if isinstance(tensor, th.Tensor):
        return tensor.cuda() if cuda else tensor
    else:
        raise RuntimeError('Please enter a pyth tensor, now a {} is received.'.format(type(tensor)))

def batchnorm(batch):
    if isinstance(batch, th.Tensor):
        return (batch - batch.mean(dim=0)) / (batch.std(dim=0) + 1e-7)
    else:
        raise RuntimeError('Please enter a pytorch tensor, now a {} is received.'.format(type(batch)))

def get_grad_norm(args, params):
    grad_norms = th.nn.utils.clip_grad_norm_(params, args.grad_clip_eps)
    return grad_norms

def merge_dict(stat, key, value):
    if key in stat.keys():
        stat[key] += value
    else:
        stat[key] = value

def n_step(rewards, last_step, done, next_values, n_step, args):
    cuda = th.cuda.is_available() and args.cuda
    returns = cuda_wrapper(th.zeros_like(rewards), cuda=cuda)
    i = rewards.size(0)-1
    while i >= 0:
        if last_step[i]:
            next_return = 0 if done[i] else next_values[i].detach()
            for j in reversed(range(i-n_step+1, i+1)):
                returns[j] = rewards[j] + args.gamma * next_return
                next_return = returns[j]
            i -= n_step
            continue
        else:
            next_return = next_values[i+n_step-1].detach()
        for j in reversed(range(n_step)):
            g = rewards[i+j] + args.gamma * next_return
            next_return = g
        returns[i] = g.detach()
        i -= 1
    return returns

def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)

def dict2str(dict, dict_name):
    
    string = [f'{dict_name}:']
    for k, v in dict.items():
        string.append(f'\t{k}: {v}' )
    string = "\n".join(string)
    return string



def plot_attention_weights_over_episode(att_weights_episode, agent_names=None):
    """
    Plot heatmaps of attention weights for each agent across time.
    
    att_weights_episode: list of agent_att_weights per step. Each is [num_agents][B, H, N-1]
    agent_names: Optional list of agent names for labeling
    """
    num_steps = len(att_weights_episode)
    num_agents = len(att_weights_episode[0])
    num_heads = att_weights_episode[0][0].shape[1]

    for agent_idx in range(num_agents):
        fig, axs = plt.subplots(num_heads, 1, figsize=(10, 2.5 * num_heads), squeeze=False)
        fig.suptitle(f"Attention Evolution for Agent {agent_names[agent_idx] if agent_names else agent_idx}")

        for h in range(num_heads):
            # Collect [time_steps, N-1] weights for head h
            weights_matrix = np.stack([
                att_weights_episode[t][agent_idx][0, h].detach().cpu().numpy()
                for t in range(num_steps)
            ])  # shape: [T, N-1]

            sns.heatmap(weights_matrix, ax=axs[h, 0], cmap='viridis', cbar=True)
            axs[h, 0].set_ylabel("Time Step")
            axs[h, 0].set_xlabel("Attended Agent")
            axs[h, 0].set_title(f"Head {h}")

        plt.tight_layout()
        plt.show()
        

def plot_mean_rewards(episode_rewards, save_path="plots/mean_rewards.png"):
    """
    Plots the mean reward per episode and saves the plot as an image.

    Parameters:
    - episode_rewards: list of mean rewards per episode
    - save_path: path to save the plot image
    """
    if not episode_rewards:
        print("No rewards to plot.")
        return

    episodes = list(range(1, len(episode_rewards) + 1)) 

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, episode_rewards, marker='o', linestyle='-', color='b', label='Mean Reward')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward per Episode')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved mean rewards plot to {save_path}")
    

def plot_attention_weights_over_episodes(attention_data, agent_idx=0, head_idx=0, save_dir="plots"):
    """
    Plots average attention weight evolution across episodes for a given agent and head,
    and saves the plot as an image.

    Parameters:
    - attention_data: List of episodes, each a list of attention weights per step.
    - agent_idx: Index of the agent.
    - head_idx: Index of the attention head.
    - save_dir: Directory where the plot will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    avg_weights_per_episode = []

    for ep in attention_data:
        weights_steps = []
        for step_weights in ep:
            attn = step_weights[agent_idx]
            if isinstance(attn, th.Tensor):
                attn = attn.detach().cpu().numpy()
            weights_steps.append(attn[0, head_idx])
        avg_weights = np.mean(weights_steps, axis=0)
        avg_weights_per_episode.append(avg_weights)

    avg_weights_per_episode = np.array(avg_weights_per_episode)

    plt.figure(figsize=(10, 5))
    for j in range(avg_weights_per_episode.shape[1]):
        plt.plot(avg_weights_per_episode[:, j], label=f'Attention to Agent {j if j < agent_idx else j+1}')
    
    plt.title(f'Attention Evolution - Agent {agent_idx} - Head {head_idx}')
    plt.xlabel('Episode')
    plt.ylabel('Attention Weight')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f"attention_evolution_agent{agent_idx}_head{head_idx}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved attention plot to {save_path}")