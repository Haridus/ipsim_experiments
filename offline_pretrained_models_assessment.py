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

#====================================================================
EnvFactory.constructors['ECSTR_S0'] = create_env_ECSTR_S0
ControlFactory.constructors['ECSTR_S0'] = create_pid_conrol_ECSTR_S0

EnvFactory.constructors['ReactorEnv'] = create_env_ReactorEnv
ControlFactory.constructors['ReactorEnv'] = create_pid_control_ReactorEnv

EnvFactory.constructors['DistillationColumn'] = create_env_DistillationColumn
ControlFactory.constructors['DistillationColumn'] = create_pid_conrol_DistillationColumn

#====================================================================      
def ECSTR_S0_show_data(algs_data):
    fig = plt.figure(figsize=(12,3))
    plt.subplot(1, 5, 1)
    for alg_name in algs_data:
        print(alg_name)
        _data = algs_data[alg_name]
        plt.plot(_data["C"], label=alg_name)
    plt.ylim(0.8,1)
    
    plt.subplot(1, 5, 2)
    for alg_name in algs_data:
        _data = algs_data[alg_name]
        plt.plot(_data["T"])
    plt.ylim(40,65)

    plt.subplot(1, 5, 3)
    for alg_name in algs_data:
        _data = algs_data[alg_name]
        plt.plot(_data["h"])
    plt.ylim(0.6,0.8)

    plt.subplot(1, 5, 4)
    for alg_name in algs_data:
        _data = algs_data[alg_name]
        plt.plot(_data["Tc"])
    plt.ylim(20,40)

    plt.subplot(1, 5, 5)
    for alg_name in algs_data:
        _data = algs_data[alg_name]
        plt.plot(_data["q"])
    plt.ylim(0.0,0.2)

    plt.gca().legend([_ for _ in algs_data])
    plt.show()

def assess_ECSTR_S0(config, algs):
    config['normalize'] = False
    config['compute_diffs_on_reward'] = False

    control = ControlFactory.create(config=config)
    env_name = config["process_name"]
    location = "offline"
    logdir = os.path.join(os.path.join(os.path.join(".","pretrained"), f"{env_name}"),f"{location}")

    algorithms = []
    algorithms.append( (control, 'baseline', config['normalize'], ) )
    for alg_name in algs :
        algorithms.append( (RLModel(alg_name, logdir), alg_name, config['normalize'], ) )

    init_state = [0.2, 25, 0.2]
    setpoints  = [0.8778252, 51.34660837, 0.659]

    processes_data = {}
    metrics_calculators = {}
    for alg, alg_name, normalize in algorithms:
        _state = deepcopy(init_state)
        _env = EnvFactory.create(config=config)
        _env.reset(initial_state = init_state)
        #_env.reset()
        processes_data[alg_name] = None
        metrics_calculators[alg_name] = MetricsCalculator(setpoints=setpoints, dt = 1)
        _mc = metrics_calculators[alg_name]

        iterations = 100
        _process_data = {"C":[], "T":[], "h":[], "Tc": [], "q":[]}
        for _ in range(iterations):
            _u = alg.predict(_state)
            _process_data["C"].append(_state[0])
            _process_data["T"].append(_state[1])
            _process_data["h"].append(_state[2])
            _process_data["Tc"].append(_u[0])
            _process_data["q"].append(_u[1])
            _mc.update(_state)

            _observation, _reward, _done, _info = _env.step(_u)
            _state = _observation
            
            processes_data[alg_name] = _process_data

        print(f"{alg_name}: process metrics: ISF={_mc.ISF()}; IAE={_mc.IAE()}; ITAE={_mc.ITAE()}; ITSH{_mc.ITSH()};")
                
    ECSTR_S0_show_data(processes_data)

def DistillationColumn_show_data(algs_data):
    fig = plt.figure(figsize=(12,3))
    plt.subplot(1, 2, 1)
    for alg_name in algs_data:
        print(alg_name)
        _data = algs_data[alg_name]
        plt.plot(_data["rrs"], label=alg_name)
    plt.ylim(2,3)
    
    plt.subplot(1, 2, 2)
    for alg_name in algs_data:
        _data = algs_data[alg_name]
        plt.plot(_data["xds"])
    plt.ylim(0.6,0.9)

    plt.gca().legend([_ for _ in algs_data])
    plt.show()


def assess_DistillationColumn(config, algs):
    config['normalize'] = False
    config['compute_diffs_on_reward'] = False
    
    control = ControlFactory.create(config=config)
    env_name = config["process_name"]
    location = "offline"
    logdir = os.path.join(os.path.join(os.path.join(".","pretrained"), f"{env_name}"),f"{location}")

    algorithms = []
    algorithms.append( (control, 'baseline', config['normalize'], ) )
    for alg_name in algs :
        algorithms.append( (RLModel(alg_name, logdir), alg_name, config['normalize'], ) )

    init_state = np.array([],dtype=np.float32)
    setpoints  = np.array([0.80,],dtype=np.float32)
        
    processes_data = {}
    metrics_calculators = {}
    for alg, alg_name, normalize in algorithms:
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
            _process_data["xds"].append(_state[0])
            _process_data["rrs"].append(_u[0])
            _mc.update(_state)

            _observation, _reward, _done, _info = _env.step(_u)
            _state = _observation
            
            processes_data[alg_name] = _process_data

        print(f"{alg_name}: process metrics: ISF={_mc.ISF()}; IAE={_mc.IAE()}; ITAE={_mc.ITAE()}; ITSH{_mc.ITSH()};")
                
    DistillationColumn_show_data(processes_data)

#======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--process', type = str, default = 'ECSTR_S0', help = 'Process model name')
    parser.add_argument('-w','--work_dir', type = str, default=os.path.dirname(__file__), help = 'Working directory')
    parser.add_argument('-a','--algs', nargs='+', default=[ 'BC'
                                                           , 'CQL'
                                                           , 'PLAS'
                                                           , 'PLASWithPerturbation'
                                                           , 'BEAR'
                                                           , 'SAC'
                                                           , 'BCQ'
                                                           , 'CRR'
                                                           , 'AWAC'
                                                           , 'DDPG'
                                                           , 'TD3'
                                                           , 'COMBO'
                                                           , 'MOPO'
                                                           ]
                                                           , help = 'list of used algorithms')    
    
    args = parser.parse_args()
    os. chdir(args.work_dir)
    project_title = args.process
    config = load_config_yaml(args.work_dir, args.process)

    if args.process == 'ECSTR_S0':
        assess_ECSTR_S0(config=config, algs=args.algs)
    if args.process == 'DistillationColumn':
        assess_DistillationColumn(config=config, algs=args.algs)