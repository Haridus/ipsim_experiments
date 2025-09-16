#================================================================
from utils.utils import *
from utils.data_generation import *
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
AlgorithmsFactory.constructors["AWAC"] = setup_alg_AWAC
AlgorithmsFactory.constructors["COMBO"] = setup_alg_COMBO
AlgorithmsFactory.constructors["MOPO"] = setup_alg_MOPO

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
    parser.add_argument('-p','--process', type = str, default = 'ReactorEnv', help = 'Process model name')
    parser.add_argument('-w','--work_dir', type = str, default=os.path.dirname(__file__), help = 'Working directory')
    parser.add_argument('-s','--seed', type = int, default=42, help = 'Seed value') 
    parser.add_argument('-a','--algs', nargs='+', default=['COMBO','MOPO','AWAC','DDPG', 'TD3', 'BC', 'CQL', 'PLAS', 'PLASWithPerturbation', 'BEAR', 'SAC', 'BCQ', 'CRR', ], help = 'list of used algorithms')    
    args = parser.parse_args()

    os.chdir(args.work_dir)
    project_title = args.process
    config = load_config_yaml(args.work_dir, args.process)

    seed = config.get('seed', args.seed)
    config['seed'] = seed
    seeds = init_seeds(config)
    env = EnvFactory.create(config=config)

    training_dataset_location = os.path.join(form_dataset_location_path(config), config['training_dataset'])
    eval_dataset_location = os.path.join(form_dataset_location_path(config), config['eval_dataset'])
    logs_location = form_logs_location_path(config)

    with open(training_dataset_location, 'rb') as f:
        training_dataset_pkl = pickle.load(f)
    with open(eval_dataset_location, 'rb') as f:
        eval_dataset_pkl = pickle.load(f)
    
    dataset = d3rlpy.dataset.MDPDataset(training_dataset_pkl['observations'], training_dataset_pkl['actions'], training_dataset_pkl['rewards'], training_dataset_pkl['terminals'])
    eval_dataset = d3rlpy.dataset.MDPDataset(eval_dataset_pkl['observations'], eval_dataset_pkl['actions'], eval_dataset_pkl['rewards'], eval_dataset_pkl['terminals'])
    feeded_episodes = dataset.episodes
    eval_feeded_episodes = eval_dataset.episodes
    config['feeded_episodes'] = feeded_episodes
    config['eval_feeded_episodes'] = eval_feeded_episodes

    seed_data = SeedData(save_path=logs_location, seeds=seeds, resume_from={
        "seed": None,
        "dataset_name": None,
        "algo_name": None,
    })

    for seed in seeds:
        d3rlpy.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        for algo_name in args.algs:
            current_positions = {
                "seed": seed,
                "algo_name": algo_name,
            }
            
            logdir = os.path.join(logs_location,str(seed))
            acutal_dir = os.path.join(logdir,algo_name)
            config['logdir'] = logdir # log dir for run --> needed for some algos
            
            wandb_run = wandb.init(reinit=True, project=project_title, name=acutal_dir, dir=logdir)

            prev_evaluate_on_environment_scorer = float('-inf')
            prev_continuous_action_diff_scorer = float('inf')
            if not seed_data.resume_checker(current_positions):
                continue
            
            algo, scorers = AlgorithmsFactory.create(algo_name, config=config)
            
            if config['evaluate_on_environment']:
                scorers['evaluate_on_environment_scorer'] = d3rlpy.metrics.scorer.evaluate_on_environment(env)
        
            for epoch, metrics in algo.fitter(feeded_episodes, eval_episodes=eval_feeded_episodes, n_epochs=config['N_EPOCHS'], with_timestamp=False, logdir=logdir, scorers=scorers):
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
            wandb_run.finish()
