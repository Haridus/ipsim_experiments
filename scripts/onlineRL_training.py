#================================================================
from utils.utils import *
from utils.constructors import *
from mgym.algorithms import *

import argparse
import wandb

import yaml
import wandb
import numpy as np
from datetime import datetime
import pandas as pd
import json
import argparse

import pandas as pd

import ray as ray 
from ray import tune
from ray.tune.registry import register_env
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
from ray.autoscaler.sdk import request_resources

#================================================================
EnvFactory.constructors['ECSTR_S0'] = create_env_ECSTR_S0
ControlFactory.constructors['ECSTR_S0'] = create_pid_conrol_ECSTR_S0

EnvFactory.constructors['ReactorEnv'] = create_env_ReactorEnv
ControlFactory.constructors['ReactorEnv'] = create_pid_control_ReactorEnv

EnvFactory.constructors['DistillationColumn'] = create_env_DistillationColumn
ControlFactory.constructors['DistillationColumn'] = create_pid_conrol_DistillationColumn

#-------------------------------------------------------------------
AlgorithmsFactory.constructors["ppo"] = setup_alg_ppo
AlgorithmsFactory.constructors["pg"] = setup_alg_pg # fail to train with error category.encode("ascii") + b"/" + key.encode("ascii"))  AttributeError: 'property' object has no attribute 'encode'
AlgorithmsFactory.constructors["ars"] = setup_alg_ars # very fast train
AlgorithmsFactory.constructors["ddpg"] = setup_alg_ddpg #fail to call predict after train
AlgorithmsFactory.constructors["apex_ddpg"] = setup_alg_apex_ddpg #fail to train with some additional dependansies
AlgorithmsFactory.constructors["a3c"] = setup_alg_a3c # failed after aprox 1000 iterations with tensor[nan nan] error
AlgorithmsFactory.constructors["sac"] = setup_alg_sac 
AlgorithmsFactory.constructors["impala"] = setup_alg_impala #failed around 500 iterations
AlgorithmsFactory.constructors["a2c"] = setup_alg_a2c # failed after aprox 1000 iterations with tensor[nan nan] error

#================================================================
def form_online_logs_location_path(config, online = False):
    env_name = config["process_name"]
    location = config.get("online_logs_location",f"ray")
    path = os.path.join(os.path.join(".", f"{env_name}"),f"{location}")
    return path

if __name__ == "__main__":
    global ACTUAL_DIR
    global ONLINE_PREV_EVALUATE_ON_ENVIRONMENT_SCORER

    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--process', type = str, default = 'DistillationColumn', help = 'Process model name')
    parser.add_argument('-w','--work_dir', type = str, default=os.path.dirname(__file__), help = 'Working directory')

    args = parser.parse_args()
    
    os. chdir(args.work_dir)
    project_title = args.process+"_online"
    config = load_config_yaml(args.work_dir, args.process)
    logs_location = os.path.abspath(form_online_logs_location_path(config, online=True))
    log_dir = os.path.abspath(os.path.join(logs_location, config['online_alg']))
    print(logs_location)
    print(log_dir)
    mkdir(logs_location)
    mkdir(log_dir)

    rl_trainer, rl_config = AlgorithmsFactory.create(config['online_alg'],config)
    wandb.tensorboard.patch(root_logdir=os.path.abspath(logs_location))
    
    env_config = {
            "env_name": config['process_name'],
            "normalize": config['normalize'],
            "dense_reward": config['dense_reward'],
            "compute_diffs_on_reward": config['compute_diffs_on_reward'],
        }

    def env_creator(env_config):
        if config['process_name'] == 'ECSTR_S0':
            return create_env_ECSTR_S0(config)
        if config['process_name'] == 'ReactorEnv':
            return create_env_ReactorEnv(config)
        if config['process_name'] == 'DistillationColumn':
            return create_env_DistillationColumn(config)
        raise Exception('unknown processs name')
    
    register_env("ECSTR_S0", env_creator)
    register_env("ReactorEnv", env_creator)
    register_env("DistillationColumn", env_creator)

    if config['use_tune']:
        with wandb.init(project = config['process_name'], dir=os.path.abspath(log_dir),   sync_tensorboard = True) as run:
            rl_config["env"] = args.process
            rl_config["num_gpus"] = config.get('num_gpus',1)
            rl_config["framework"] = "torch"
            rl_config["num_workers"] = config.get('num_workers',1)
            rl_config["evaluation_num_workers"] = config.get('num_workers',1)
            #rl_config["num_cpus_per_worker"] = 4
            rl_config["evaluation_interval"] = int(config['train_iter'] / 10)
            #rl_config["evaluation_duration"] = 10
            rl_config["env_config"] = env_config
            rl_config["logger_config"] = {
                "wandb": {
                    "project": config['process_name'],
                    "log_config": True, 
                }
            }
            if config['scheduler_name'] == 'asha_scheduler':
                scheduler = tune.schedulers.ASHAScheduler(
                    time_attr='training_iteration',
                    metric='episode_reward_mean',
                    mode='max',
                    max_t=config['train_iter'],
                    grace_period=10,
                    reduction_factor=3,
                    brackets=1
                )
                analysis = tune.run(rl_trainer, 
                    metric='episode_reward_mean',
                    mode='max',
                    time_budget_s=config['time_budget_s'],
                    config=rl_config, 
                    local_dir=log_dir,
                    log_to_file='logfile.log',
                    checkpoint_freq=1,
                    checkpoint_at_end=True,
                    keep_checkpoints_num=5,
                    checkpoint_score_attr="episode_reward_mean", # Specifies by which attribute to rank the best checkpoint. Default is increasing order. If attribute starts with min- it will rank attribute in decreasing order, i.e. min-validation_loss.
                    stop={"training_iteration": config['train_iter']},
                    scheduler=scheduler,
                    loggers=DEFAULT_LOGGERS #+ (WandbLogger, )
                )
            elif config['scheduler_name'] == 'fifo_scheduler':
                analysis = tune.run(rl_trainer, 
                    metric='episode_reward_mean',
                    mode='max',
                    time_budget_s=config['time_budget_s'],
                    config=rl_config, 
                    #resources_per_trial = plFactory,
                    #resources_per_trial={"cpu": 4, "gpu": 4}, 
                    local_dir=log_dir,
                    log_to_file='logfile.log',
                    checkpoint_freq=1,
                    checkpoint_at_end=True,
                    keep_checkpoints_num=5,
                    checkpoint_score_attr="episode_reward_mean", # Specifies by which attribute to rank the best checkpoint. Default is increasing order. If attribute starts with min- it will rank attribute in decreasing order, i.e. min-validation_loss.
                    stop={"training_iteration": config['train_iter']},
                    loggers=DEFAULT_LOGGERS #+ (WandbLogger, )
                )
            else:
                raise NotImplementedError
        
        print("Best config: ", analysis.get_best_config(
            metric="episode_reward_mean", mode="max"
        ))
        
    else:
        CHECKPOINT_ROOT = os.path.join(log_dir,datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        best_episode_reward_mean = -1e8
        checkpoint_file = ""
        info = ray.init(ignore_reinit_error=True, num_gpus=config.get('num_gpus',4))
        status = "reward {:6.2f} {:6.2f} {:6.2f}  len {:4.2f}  saved {}"
        rl_config["num_gpus"] = config.get('num_gpus',4)
        rl_config["env_config"] = env_config
        rl_config["framework"] = "torch"
        rl_config["num_workers"] = config.get('num_workers',4) 
        rl_config["evaluation_num_workers"] = 1
        rl_config["evaluation_interval"] = int(config['train_iter'] / 10)
        #rl_config["evaluation_duration"] = 10
        agent = rl_trainer(rl_config, env="my_env")
        df = pd.DataFrame(columns=[ "min_reward", "avg_reward", "max_reward", "steps", "checkpoint"])
        for i in range(config['train_iter']):
            result = agent.train()
            checkpoint_file = agent.save(CHECKPOINT_ROOT)
            if result["episode_reward_mean"] > best_episode_reward_mean:
                best_episode_reward_mean = result["episode_reward_mean"]
                best_iter = i
                best_checkpoint_file = checkpoint_file
            row = [
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                checkpoint_file,
            ]
            df.loc[len(df)] = row
            print(status.format(*row))
        df.to_csv(f"{CHECKPOINT_ROOT}/{config['model_name']}_results.csv")
        result_dict = {"best_episode_reward_mean": best_episode_reward_mean, "best_iter": best_iter, "checkpoint_file": best_checkpoint_file}
        json.dump(result_dict, open(f"{CHECKPOINT_ROOT}/{config['model_name']}_result.json", 'w+'))     
