import torch as th
import os
import argparse

from tensorboardX import SummaryWriter

from models.model_registry import Model, Strategy
from environemet.Reactorenv import ReactorEnv
from utilities.util import convert, dict2str,plot_mean_rewards, plot_attention_weights_over_episodes
from utilities.trainer import DPGTrainer



parser = argparse.ArgumentParser(description="Train rl agent.")
parser.add_argument("--save-path", type=str, nargs="?", default="./", help="Please enter the directory of saving model.")
parser.add_argument("--alg", type=str, nargs="?", default="ATT_maddpg", help="Please enter the alg name.")

argv = parser.parse_args()



alg_config_dict = {
    "episodic":False,
    "replay":True,
    "policy_lrate":1e-4,
    "critic_lrate":1e-4,
    "mixer":False,
    "continuous": True,
    "deterministic": True,
    "eval_freq": 1000,
    "hid_size":64,
    "critic_hid_size":64,
    "target": True,
    "sa_sizes": [(2, 1), (2, 1), (2, 1), (2, 1)],
    "layernorm": True,
    "hid_activation": "relu",
    "norm_input": True,
    "norm_in":False,
    "save_model_freq":2,
    "train_episodes_num":2
    ,
    "max_steps": 30,
    "agent_type": "rnn",
    "shared_params":False,
    "target_lr":0.001,
    "behaviour_update_freq":10,
    "policy_update_epochs": 4,
    "value_update_epochs": 5,
    "batch_size":12,
    "replay_warmup":10,
    "init_type" :"normal",
    "agent_id": False,
    "attend_heads": 2,
    "init_std": 0.1,
    "replay_buffer_size": 15,
    "value_lrate": 1e-4,
    "entr":False,
    "reward_normalisation":False,
    "gamma": 0.99,
    "setpoint_change_freq": 1000,
    "target_update_freq":4,
    "gaussian_policy": False,
    "grad_clip_eps": 0.5,
        

}

# define envs

env_config_dict=dict(
    A=1.0,
    rho_a=1.0,
    rho_b=1.0,
    rho_w=1.0,
    r=1.0,
    ko=1.0,
    E=1.0,
    R=1.0,
    cp_w=1.0,
    cp_a=1.0,
    cp_b=1.0,
    dHr=1.0,
    Tref=1.0,
    Cai=1.0,
    Cbi=1.0,
    h_sp=3.0,
    Ca_sp=1.0,
    Cb_sp=1.0,
    Cc_sp=50.0,
    T_sp=320.0,
    h_min=1.0,
    h_max=4.0,
    T_min=298.0,
    T_max=450.0,
    dt=1.0,
    max_steps=30,
    initial_state=[1.0, 1.0, 1.0, 1.0, 1.0],

)

nmpc_env=env_config_dict
nmpc_env["reward_type"] = "nmpc"

dir_env=env_config_dict
dir_env["reward_type"] = "derivative"

norm_env=env_config_dict
norm_env["reward_type"] = "normal"






normal_env = ReactorEnv(norm_env);
nmpc_env=ReactorEnv(nmpc_env);
dir_env = ReactorEnv(dir_env);


alg_config_dict["agent_num"] = 4
alg_config_dict["obs_dims"] = [2]*alg_config_dict["agent_num"]
alg_config_dict["action_dim"] = 1

args = convert(alg_config_dict)


log_name = argv.alg + "_" + "Reactor" + "_" + str(args.agent_num) + "_agents_" + alg_config_dict ["agent_type"] + "_" +env_config_dict["reward_type"] 

# define the save path
if argv.save_path[-1] == "/":
    save_path = argv.save_path
else:
    save_path = argv.save_path+"/"

# create the save folders
if "model_save" not in os.listdir(save_path):
    os.mkdir(save_path + "model_save")
if "tensorboard" not in os.listdir(save_path):
    os.mkdir(save_path + "tensorboard")
if log_name not in os.listdir(save_path + "model_save/"):
    os.mkdir(save_path + "model_save/" + log_name)
if log_name not in os.listdir(save_path + "tensorboard/"):
    os.mkdir(save_path + "tensorboard/" + log_name)
else:
    path = save_path + "tensorboard/" + log_name
    for f in os.listdir(path):
        file_path = os.path.join(path,f)
        if os.path.isfile(file_path):
            os.remove(file_path)

# create the logger
logger = SummaryWriter(save_path + "tensorboard/" + log_name)

model = Model[argv.alg]

strategy = Strategy[argv.alg]




nmpc_train = DPGTrainer(args, model,nmpc_env, logger)
normal_env_train = DPGTrainer(args, model, normal_env, logger)
dir_train = DPGTrainer(args, model, dir_env, logger)


with open(save_path + "tensorboard/" + log_name + "/log.txt", "w+") as file:
    alg_args2str = dict2str(alg_config_dict, 'alg_params')
    env_args2str = dict2str(env_config_dict, 'env_params')
    file.write(alg_args2str + "\n")
    file.write(env_args2str + "\n")
mean_reawads=[]
all_episode_attention_weights = []
for i in range(args.train_episodes_num):
    
    
    
    print(f"=======Epesoide {i+1}/{args.train_episodes_num}=======")
    
    stat = {}
    stat =nmpc_train.run(stat, i)
    
    mean_reawads.append(stat["mean_train_reward"].item())
    if 'ep_attention_weights' in stat:
        # Each element is a list of [agents][heads][weights] over timesteps
        all_episode_attention_weights.append(stat['ep_attention_weights'])

    
    if i%args.save_model_freq == args.save_model_freq-1:
      
        th.save({"model_state_dict":nmpc_train.behaviour_net.state_dict()}, save_path + "model_save/" + log_name + f"/model{i+1}.pt")
        print ("The model is saved!\n")
    print(f"=======Epesoide {i+1}/{args.train_episodes_num}=======")
print ("the epesoide mean rewards ",mean_reawads)
plot_mean_rewards(mean_reawads )
plot_attention_weights_over_episodes(all_episode_attention_weights, agent_idx=0, head_idx=0)
logger.close()
