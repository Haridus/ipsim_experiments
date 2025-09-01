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
AlgorithmsFactory.constructors["BEAR"] = setup_alg_BEAR
AlgorithmsFactory.constructors["BCQ"] = setup_alg_BCQ
AlgorithmsFactory.constructors["CRR"] = setup_alg_CRR
AlgorithmsFactory.constructors["AWR"] = setup_alg_AWR
AlgorithmsFactory.constructors["AWAC"] = setup_alg_AWAC
AlgorithmsFactory.constructors["AWAC"] = setup_alg_AWAC
AlgorithmsFactory.constructors["AWAC"] = setup_alg_MOPO

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

    with open(config['training_dataset_loc'], 'rb') as f:
        training_dataset_pkl = pickle.load(f)
    with open(config['eval_dataset_loc'], 'rb') as f:
        eval_dataset_pkl = pickle.load(f)
    
    seed_data = SeedData(save_path=config['default_loc'], resume_from={
        "seed": None,
        "dataset_name": None,
        "algo_name": None,
    })

    algo_names = args.algs
    env = EnvFactory.create(config=config)

    for seed in seeds:
        d3rlpy.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
        dataset = d3rlpy.dataset.MDPDataset(training_dataset_pkl['observations'], training_dataset_pkl['actions'], training_dataset_pkl['rewards'], training_dataset_pkl['terminals'])
        eval_dataset = d3rlpy.dataset.MDPDataset(eval_dataset_pkl['observations'], eval_dataset_pkl['actions'], eval_dataset_pkl['rewards'], eval_dataset_pkl['terminals'])
        feeded_episodes = dataset.episodes
        eval_feeded_episodes = eval_dataset.episodes
        config['feeded_episodes'] = feeded_episodes
        config['eval_feeded_episodes'] = eval_feeded_episodes
        
         
        for algo_name in algo_names:
            current_positions = {
                "seed": seed,
                "algo_name": algo_name,
            }
            config['logdir'] = config['default_loc']+str(seed)
            acutal_dir = config['logdir']+'/'+algo_name

            prev_evaluate_on_environment_scorer = float('-inf')
            prev_continuous_action_diff_scorer = float('inf')
            global ONLINE_PREV_EVALUATE_ON_ENVIRONMENT_SCORER
            ONLINE_PREV_EVALUATE_ON_ENVIRONMENT_SCORER = float('-inf')
            if not seed_data.resume_checker(current_positions):
                continue
            
            algo, scorers = AlgorithmsFactory.create(config=config)
            
            if config['evaluate_on_environment']:
                scorers['evaluate_on_environment_scorer'] = d3rlpy.metrics.scorer.evaluate_on_environment(env)
        
            for epoch, metrics in algo.fitter(feeded_episodes, eval_episodes=eval_feeded_episodes, n_epochs=config['N_EPOCHS'], with_timestamp=False, logdir=config['logdir'], scorers=scorers):
                wandb.log(metrics)
                if config['evaluate_on_environment']:
                    if metrics['evaluate_on_environment_scorer'] > prev_evaluate_on_environment_scorer:
                        prev_evaluate_on_environment_scorer = metrics['evaluate_on_environment_scorer']
                        algo.save_model(os.path.join(acutal_dir, 'best_evaluate_on_environment_scorer.pt'))
                if metrics['continuous_action_diff_scorer'] < prev_continuous_action_diff_scorer:
                    prev_continuous_action_diff_scorer = metrics['continuous_action_diff_scorer']
                    algo.save_model(os.path.join(acutal_dir, 'best_continuous_action_diff_scorer.pt'))
            
            if config['evaluate_on_environment']:
                shutil.copyfile(os.path.join(acutal_dir, 'best_evaluate_on_environment_scorer.pt'), os.path.join(acutal_dir, 'best.pt'))
            else:
                shutil.copyfile(os.path.join(acutal_dir, 'best_continuous_action_diff_scorer.pt'), os.path.join(acutal_dir, 'best.pt'))
            wandb.wandb_run.finish()
