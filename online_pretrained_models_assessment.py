#====================================================================
from utils.utils import *
from utils.data_generation import *
from utils.constructors import *
from mgym.algorithms import *

import argparse
import codecs
import copy
import csv
from copy import deepcopy

import ray.rllib.agents.ppo as ppo
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

import ray as ray 
from ray import tune
from ray.tune.registry import register_env
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
import pandas as pd

from copy import deepcopy

#====================================================================
EnvFactory.constructors['ECSTR_S0'] = create_env_ECSTR_S0
ControlFactory.constructors['ECSTR_S0'] = create_pid_conrol_ECSTR_S0

EnvFactory.constructors['ReactorEnv'] = create_env_ReactorEnv
ControlFactory.constructors['ReactorEnv'] = create_pid_control_ReactorEnv

EnvFactory.constructors['DistillationColumn'] = create_env_DistillationColumn
ControlFactory.constructors['DistillationColumn'] = create_pid_conrol_DistillationColumn

#------------------------------------------------------
AlgorithmsFactory.constructors["ppo"] = setup_alg_ppo
AlgorithmsFactory.constructors["pg"] = setup_alg_pg
AlgorithmsFactory.constructors["ars"] = setup_alg_ars
AlgorithmsFactory.constructors["ddpg"] = setup_alg_ddpg
AlgorithmsFactory.constructors["apex_ddpg"] = setup_alg_apex_ddpg
AlgorithmsFactory.constructors["a3c"] = setup_alg_a3c
AlgorithmsFactory.constructors["sac"] = setup_alg_sac
AlgorithmsFactory.constructors["impala"] = setup_alg_impala
AlgorithmsFactory.constructors["a2c"] = setup_alg_a2c

#====================================================================
def show_metrics_env_ECSTR_S0_ReactorEnv(algs_data):
    fig = plt.figure(figsize=(12,3))
    plt.subplot(1, 5, 1)
    for alg_name in algs_data:
        print(alg_name)
        _data = algs_data[alg_name]
        plt.plot(_data["Cs"], label=alg_name)
    
    plt.subplot(1, 5, 2)
    for alg_name in algs_data:
        _data = algs_data[alg_name]
        plt.plot(_data["Ts"])
    
    plt.subplot(1, 5, 3)
    for alg_name in algs_data:
        _data = algs_data[alg_name]
        plt.plot(_data["hs"])
    
    plt.subplot(1, 5, 4)
    for alg_name in algs_data:
        _data = algs_data[alg_name]
        plt.plot(_data["Tcs"])
    
    plt.subplot(1, 5, 5)
    for alg_name in algs_data:
        _data = algs_data[alg_name]
        plt.plot(_data["qs"])

    plt.gca().legend([_ for _ in algs_data])
    plt.show()

def case_env_ECSTR_S0_ReactorEnv(algorithms, config):
    init_state = np.array([0.2, 25, 0.2],dtype=np.float32)
    setpoints  = np.array([0.8778252, 51.34660837, 0.659],dtype=np.float32)
        
    processes_data = {}
    metrics_calculators = {}
    for alg, alg_name in algorithms:
        _state = deepcopy(init_state)
        _env = EnvFactory.create(config=config)
        _env.reset(initial_state = init_state)
        processes_data[alg_name] = None
        metrics_calculators[alg_name] = MetricsCalculator(setpoints=setpoints, dt = 1)
        _mc = metrics_calculators[alg_name]
            
        _state = _env.normalize_observations(_state)
        iterations = 100
        _process_data = {"Cs":[], "Ts":[], "hs":[], "Tcs": [], "qs":[]}
        for _ in range(iterations):
            _u = alg.predict(_state)
            dn_u = _env.denormalize_actions(_u)
            dn_s = _env.denormalize_observations(_state)
            _mc.update(dn_s)

            _process_data["Cs"].append(dn_s[0])
            _process_data["Ts"].append(dn_s[1])
            _process_data["hs"].append(dn_s[2])
            _process_data["Tcs"].append(dn_u[0])
            _process_data["qs"].append(dn_u[1])                
            processes_data[alg_name] = _process_data

            _observation, _reward, _done, _info = _env.step(_u)
            _state = _observation

    return processes_data, metrics_calculators


def show_env_DistillationColumn(algs_data):
    fig = plt.figure(figsize=(12,3))
    plt.subplot(1, 2, 1)
    for alg_name in algs_data:
        print(alg_name)
        _data = algs_data[alg_name]
        plt.plot(_data["rrs"], label=alg_name)
    
    plt.subplot(1, 2, 2)
    for alg_name in algs_data:
        _data = algs_data[alg_name]
        plt.plot(_data["xds"])

    plt.gca().legend([_ for _ in algs_data])
    plt.show()

def case_env_DistillationColumn(algorithms, config):
    init_state = np.array([],dtype=np.float32)
    setpoints  = np.array([0.80,],dtype=np.float32)
        
    processes_data = {}
    metrics_calculators = {}
    for alg, alg_name in algorithms:
        _state = deepcopy(init_state)
        _env = EnvFactory.create(config=config)
        _env.reset(initial_state = init_state)
    
        processes_data[alg_name] = None
        metrics_calculators[alg_name] = MetricsCalculator(setpoints=setpoints, dt = 1)
        _mc = metrics_calculators[alg_name]

        iterations = 100    
        _state, _reward, _done, _info = _env.step(action=_env.normalize_actions([3,]))
        _process_data = {"rrs":[], "xds":[], }
        for _ in range(iterations):
            _u = alg.predict(_state)
            dn_u = _env.denormalize_actions(_u)
            dn_s = _env.denormalize_observations(_state)
            _mc.update(dn_s)

            _process_data["xds"].append(dn_s[0])
            _process_data["rrs"].append(dn_u[0])               
            processes_data[alg_name] = _process_data

            _observation, _reward, _done, _info = _env.step(_u)
            _state = _observation

    return processes_data, metrics_calculators

#======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--process', type = str, default = 'DistillationColumn', help = 'Process model name')
    parser.add_argument('-w','--work_dir', type = str, default=os.path.dirname(__file__), help = 'Working directory')
    parser.add_argument('-a','--algs', nargs='+', default=[
                                                            'ppo'
                                                          , 'sac'
                                                          , 'a2c'
                                                          , 'ars'
                                                          , 'impala'
                                                          #, 'a3c'
                                                          ], help = 'list of used algorithms')    
    args = parser.parse_args()

    os. chdir(args.work_dir)
    project_title = args.process
    config = load_config_yaml(args.work_dir, args.process)
    env_name = config["process_name"]
    config['normalize'] = True
    config['compute_diffs_on_reward'] = False
    logdir = os.path.join(os.path.join(os.path.join(".","pretrained"), f"{env_name}"),f"online")
    #wandb.tensorboard.patch(root_logdir=logdir)

    processes_data = {}
    metrics_calculators = {}
    for alg_name in args.algs:
        try:
            rl_trainer, rl_config = AlgorithmsFactory.create(alg_name=alg_name,config = config)
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

            checkpoint_path = os.path.join(logdir,alg_name)
            checkpoint_path = os.path.join(checkpoint_path,'best')
            checkpoint_path = os.path.join(checkpoint_path,'best') 
            print(checkpoint_path)

            rl_config["env_config"] = env_config
            rl_config["framework"] = "torch"
            rl_config["evaluation_interval"] = int(config['train_iter'] / 10)
            agent = rl_trainer(rl_config, env=config['process_name'])
            agent.restore(checkpoint_path)

            algorithms=[]
            algorithms.append((RayAgentWrapper(agent=agent),alg_name,))

            if args.process == 'DistillationColumn':
                _process_data, _metrics_calculators = case_env_DistillationColumn(algorithms=algorithms, config=config)
            if args.process in ['ECSTR_S0','ReactorEnv']:
                _process_data, _metrics_calculators = case_env_ECSTR_S0_ReactorEnv(algorithms=algorithms, config=config)

            processes_data[alg_name] = _process_data[alg_name]
            metrics_calculators[alg_name] = _metrics_calculators[alg_name]
        
        except Exception as exception:
            print(f'{alg_name} exception: {exception}')

    for alg_name in metrics_calculators:
        _mc = metrics_calculators[alg_name]
        print(f"{alg_name}[metrics]: {_mc.ISF()}; {_mc.IAE()}; {_mc.ITAE()}; {_mc.ITSH()};")

    if args.process == 'DistillationColumn':
        show_env_DistillationColumn(processes_data)
    if args.process in ['ECSTR_S0','ReactorEnv']:
        show_metrics_env_ECSTR_S0_ReactorEnv(processes_data)   
