#====================================================================
from utils.utils import *
from utils.data_generation import *
from utils.constructors import *
from mgym.algorithms import *

import argparse
import codecs
import copy
import csv

#====================================================================
EnvFactory.constructors['ECSTR_S0'] = create_env_ECSTR_S0
ControlFactory.constructors['ECSTR_S0'] = create_pid_conrol_ECSTR_S0

EnvFactory.constructors['ReactorEnv'] = create_env_ReactorEnv
ControlFactory.constructors['ReactorEnv'] = create_pid_control_ReactorEnv

#====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--process', type = str, default = 'ReactorEnv', help = 'Process model name')
    parser.add_argument('-w','--work_dir', type = str, default=os.path.dirname(__file__), help = 'Working directory')
    parser.add_argument('-s','--seed', type = int, default=42, help = 'Seed value') 
    parser.add_argument('-n','--num_of_initial_states', type = int, default=1000, help = 'number of initial states') 
    parser.add_argument('-a','--algs', nargs='+', default=['BC', 'CQL', 'PLAS', 'PLASWithPerturbation', 'BEAR', 'SAC', 'BCQ', 'CRR', 'AWAC', 'DDPG', 'TD3', 'COMBO', 'MOPO'], help = 'list of used algorithms')    
    args = parser.parse_args()

    os. chdir(args.work_dir)
    project_title = args.process
    config = load_config_yaml(args.work_dir, args.process)
    seed = config.get('seed', args.seed)
    
    env = EnvFactory.create(config=config)
    control = ControlFactory.create(config=config)
   
    logs_location = form_logs_location_path(config)
    logdir = os.path.join(logs_location,str(seed))
    num_episodes = args.num_of_initial_states
    
    algorithms = []
    algorithms.append( (control, 'baseline', config['normalize',] ) )
    for alg_name in args.algs :
        algorithms.append( (RLModel(alg_name, logdir), alg_name, config['normalize'], ) )

    results_csv = ['algo_name', 'on_episodes_reward_mean', 'episodes_reward_std', 'all_reward_mean', 'all_reward_std']
    for  alg, alg_name, normalize in algorithms:
        save_dir = os.path.join(config['plt_dir'], alg_name)
        observations_list, actions_list, rewards_list = env.evalute_algorithms(algorithms, num_episodes=num_episodes, initial_states=None, to_plt=False, plot_dir=save_dir)
        
        results_dict = env.report_rewards(rewards_list, algo_names=env.algorithms_to_algo_names(algorithms), save_dir=save_dir)
        results_csv.append([alg_name, results_dict[f'{alg_name}_on_episodes_reward_mean'], results_dict[f'{alg_name}_on_episodes_reward_std'], results_dict[f'{alg_name}_all_reward_mean'], results_dict[f'{alg_name}_all_reward_std']])
        np.save(os.path.join(save_dir, f'observations.npy'), observations_list)
        np.save(os.path.join(save_dir, f'actions.npy'), actions_list)
        np.save(os.path.join(save_dir, f'rewards.npy'), rewards_list)

    with codecs.open(os.path.join(config['plt_dir'], "total_results_dict.csv"), "w+", encoding="utf-8") as fp:
        csv_writer = csv.writer(fp)
        for row in results_csv:
            csv_writer.writerow(row)   

