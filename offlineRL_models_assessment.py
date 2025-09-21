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

#====================================================================
class MetricsCalculator:
    def __init__(self, setpoints, dt = 1):
        self.setpoints = setpoints
        self.dt = dt

        self.t = 0
        self.error     = [0 for x in range(len(self.setpoints))]
        self.error2     = [0 for x in range(len(self.setpoints))]
        self.sums_ISF  = [0 for x in range(len(self.setpoints))]
        self.sums_IAE  = [0 for x in range(len(self.setpoints))]
        self.sums_ITAE = [0 for x in range(len(self.setpoints))]
        self.sums_ITSH = [0 for x in range(len(self.setpoints))]
    
    def update(self, state):
        self.t += self.dt
        for _ in range(min(len(self.setpoints), len(state))):
            e = self.setpoints[_] - state[_]
            self.error[_] += e
            self.sums_ISF[_]  += e**2
            self.sums_IAE[_]  += abs(e)
            self.sums_ITAE[_] += abs(e)*self.t
            self.sums_ITSH[_] += (e**2)*self.t

    def ISF(self):
        return [sum*self.dt/2 for sum in self.sums_ISF]
    
    def IAE(self):
        return [sum*self.dt/2 for sum in self.sums_IAE]
    
    def ITAE(self):
        return [sum*self.dt/2 for sum in self.sums_ITAE]
    
    def ITSH(self):
        return [sum*self.dt/2 for sum in self.sums_ITSH]
    
def show_metrics(data, alg_name):
    plt.figure(figsize=(12,3))
    plt.title(alg_name)
    plt.subplot(1, 5, 1)
    plt.plot(data["Cs"])
    
    plt.subplot(1, 5, 2)
    plt.plot(data["Ts"])
    
    plt.subplot(1, 5, 3)
    plt.plot(data["hs"])
    
    plt.subplot(1, 5, 4)
    plt.plot(data["Tcs"])

    plt.subplot(1, 5, 5)
    plt.plot(data["qs"])
    
    plt.show()
    
def show_metrics_2(algs_data):
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
    
#======================================================================
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
    algorithms.append( (control, 'baseline', config['normalize'], ) )
    for alg_name in args.algs :
        algorithms.append( (RLModel(alg_name, logdir), alg_name, config['normalize'], ) )

    """
    results_csv = ['algo_name', 'on_episodes_reward_mean', 'episodes_reward_std', 'all_reward_mean', 'all_reward_std']
    for  alg, alg_name, normalize in algorithms:
        save_dir = os.path.join(form_plt_location_path(config), alg_name)
        alg_inf = [(alg, alg_name, normalize, )]
        observations_list, actions_list, rewards_list = env.evalute_algorithms(alg_inf, num_episodes=num_episodes, initial_states=None, to_plt=False, plot_dir=save_dir)
        
        results_dict = env.report_rewards(rewards_list, algo_names=env.algorithms_to_algo_names(alg_inf), save_dir=save_dir)
        results_csv.append([alg_name, results_dict[f'{alg_name}_on_episodes_reward_mean'], results_dict[f'{alg_name}_on_episodes_reward_std'], results_dict[f'{alg_name}_all_reward_mean'], results_dict[f'{alg_name}_all_reward_std']])
        np.save(os.path.join(save_dir, f'observations.npy'), observations_list)
        np.save(os.path.join(save_dir, f'actions.npy'), actions_list)
        np.save(os.path.join(save_dir, f'rewards.npy'), rewards_list)

    with codecs.open(os.path.join(form_plt_location_path(config), "total_results_dict.csv"), "w+", encoding="utf-8") as fp:
        csv_writer = csv.writer(fp)
        for row in results_csv:
            csv_writer.writerow(row)   
    """

    #============================================================
    if True:
        init_state = [0.1, 1, 0.1]
        setpoints  = [0.8778252, 0.659]
        
        processes_data = {}
        metrics_calculators = {}
        for alg, alg_name, normalize in algorithms:
            _state = deepcopy(init_state)
            _env = EnvFactory.create(config=config)
            _env.reset()
            processes_data[alg_name] = None
            metrics_calculators[alg_name] = MetricsCalculator(setpoints=setpoints, dt = 1)
            _mc = metrics_calculators[alg_name]

            iterations = 100
            _process_data = {"Cs":[], "Ts":[], "hs":[], "Tcs": [], "qs":[]}
            for _ in range(iterations):
                _u = alg.predict(_state)
                _observation, _reward, _done, _info = _env.step(_u)
                _state = _observation
                _mc.update((_state[0], _state[2]))
                _process_data["Cs"].append(_state[0])
                _process_data["Ts"].append(_state[1])
                _process_data["hs"].append(_state[2])
                _process_data["Tcs"].append(_u[0])
                _process_data["qs"].append(_u[1])
                processes_data[alg_name] = _process_data

            #print(f"{alg_name}: process metrics: {_mc.ISF()}; {_mc.IAE()}; {_mc.ITAE()}; {_mc.ITSH()};")
            #show_metrics(_process_data, alg_name)    
        show_metrics_2(processes_data)    
