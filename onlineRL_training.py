#================================================================
from utils.utils import *
from utils.constructors import *
from mgym.algorithms import *

import argparse
import pickle
import d3rlpy
import wandb
import shutil

#================================================================
EnvFactory.constructors['ECSTR_S0'] = create_env_ECSTR_S0
ControlFactory.constructors['ECSTR_S0'] = create_pid_conrol_ECSTR_S0

EnvFactory.constructors['ReactorEnv'] = create_env_ReactorEnv
ControlFactory.constructors['ReactorEnv'] = create_pid_control_ReactorEnv

AlgorithmsFactory.constructors["CQL"] = setup_alg_CQL
AlgorithmsFactory.constructors["PLAS"] = setup_alg_PLAS
AlgorithmsFactory.constructors["PLASWithPerturbation"] = setup_alg_PLASWithPerturbation
AlgorithmsFactory.constructors["DDPG"] = setup_alg_DDPG
AlgorithmsFactory.constructors["BC"] = setup_alg_BC
AlgorithmsFactory.constructors["TD3"] = setup_alg_TD3
AlgorithmsFactory.constructors["BEAR"] = setup_alg_BEAR
AlgorithmsFactory.constructors["SAC"] = setup_alg_SAC
AlgorithmsFactory.constructors["BCQ"] = setup_alg_BCQ
AlgorithmsFactory.constructors["CRR"] = setup_alg_CRR
AlgorithmsFactory.constructors["AWR"] = setup_alg_AWR
AlgorithmsFactory.constructors["AWAC"] = setup_alg_AWAC
AlgorithmsFactory.constructors["COMBO"] = setup_alg_COMBO
AlgorithmsFactory.constructors["MOPO"] = setup_alg_MOPO

#----------------------------------------------------------
def online_saving_callback(algo, epoch, total_step):
    mean_env_ret = d3rlpy.metrics.evaluate_on_environment(env, n_trials=10, epsilon=0.0)(algo)
    global ACTUAL_DIR
    global ONLINE_PREV_EVALUATE_ON_ENVIRONMENT_SCORER
    if mean_env_ret < ONLINE_PREV_EVALUATE_ON_ENVIRONMENT_SCORER:
        ONLINE_PREV_EVALUATE_ON_ENVIRONMENT_SCORER = mean_env_ret
        algo.save_model(os.path.join(ACTUAL_DIR, 'best_env.pt'))

#================================================================
def init_seeds(config):
    seeds = None
    seed = config.get('seed', None)
    if seed is not None:
        seeds = [seed]
    else:
        num_of_seeds = config['num_of_seeds']
        seeds = []
        for i in range(num_of_seeds):
            seeds.append(random.randint(0, 2**32-1))
    return seeds

if __name__ == "__main__":
    global ACTUAL_DIR
    global ONLINE_PREV_EVALUATE_ON_ENVIRONMENT_SCORER

    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--process', type = str, default = 'ECSTR_S0', help = 'Process model name')
    parser.add_argument('-w','--work_dir', type = str, default=os.path.dirname(__file__), help = 'Working directory')
    parser.add_argument('-s','--seed', type = int, default=None, help = 'Seed value') 
    parser.add_argument('-a','--algs', nargs='+', default=['BC', 'CQL', 'PLAS', 'PLASWithPerturbation', 'BEAR', 'SAC', 'BCQ', 'CRR', 'AWAC', 'DDPG', 'TD3', 'COMBO', 'MOPO'], help = 'list of used algorithms')    
    args = parser.parse_args()
    
    os. chdir(args.work_dir)
    project_title = args.process
    config = load_config_yaml(args.work_dir, args.process)
    seed = config.get('seed', args.seed)
    config['seed'] = seed
    seeds = init_seeds(config)

    seed_data = SeedData(save_path=config['default_loc'], resume_from={
        "seed": None,
        "dataset_name": None,
        "algo_name": None,
    })

    env = EnvFactory.create(config=config)

    for seed in seeds:
        d3rlpy.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        for algo_name in args.algs:
            current_positions = {
                "seed": seed,
                "algo_name": algo_name,
            }
            config['logdir'] = config['default_loc']+"_ONLINE_"+str(seed)
            acutal_dir = config['logdir']+'/'+algo_name

            ONLINE_PREV_EVALUATE_ON_ENVIRONMENT_SCORER = float('-inf')
            ACTUAL_DIR = acutal_dir

            if not seed_data.resume_checker(current_positions):
                continue
            
            algo, scorers = AlgorithmsFactory.create(config=config)
            
            if config['evaluate_on_environment']:
                scorers['evaluate_on_environment_scorer'] = d3rlpy.metrics.scorer.evaluate_on_environment(env)

            explorer = d3rlpy.online.explorers.LinearDecayEpsilonGreedy(start_epsilon=config['explorer_start_epsilon'], end_epsilon=config['explorer_end_epsilon'], duration=config['explorer_duration'])
            buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=config['buffer_maxlen'], env=env)
            
            algo.fit_online(  env, buffer
                            , explorer=explorer
                            , eval_env=env
                            , n_steps=config['N_EPOCHS']*config['n_steps_per_epoch']
                            , n_steps_per_epoch=['n_steps_per_epoch']
                            , update_interval=['online_update_interval']
                            , random_steps=['online_random_steps']
                            , save_interval=['online_save_interval']
                            , with_timestamp=False
                            , tensorboard_dir=config['logdir']+'/tensorboard'
                            , logdir=config['logdir']
                            , callback=online_saving_callback)
            wandb.wandb_run.finish()
