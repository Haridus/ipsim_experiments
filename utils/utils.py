import os
import yaml

#=====================================================================
class EnvFactory:
    constructors = { }    
    @staticmethod
    def create(config):
        env_name = config['process_name']
        if env_name in EnvFactory.constructors:
            return EnvFactory.constructors[env_name](config=config)
        raise Exception("Unknown env")

class ControlFactory:
    constructors = { }

    @staticmethod
    def create(config):
        env_name = config['process_name']
        if env_name in ControlFactory.constructors:
            return ControlFactory.constructors[env_name](config=config)
        raise Exception("Unknown env")
    
class AlgorithmsFactory:
    constructors = { }

    @staticmethod
    def create(alg_name, config):
        if alg_name in AlgorithmsFactory.constructors:
            return AlgorithmsFactory.constructors[alg_name](config=config)
        raise Exception("Unknown algorithm")

#----------------------------------------------------------------------
def load_config_yaml(work_dir, model_name):
    config_path = f'{work_dir}/{model_name}.yaml'
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)
        config['work_dir'] = work_dir
        return config
    return {}

def mkdir(path):
    result = False
    try:
        os.makedirs(path, exist_ok=True)
        result = True
    except OSError as e:
        pass
    return result

def form_logs_location_path(config, online = False):
    env_name = config["process_name"]
    location = config.get("logs_location",f"d3rlpy_logs")
    path = os.path.join(os.path.join(".", f"{env_name}"),f"{location}")
    path = os.path.join(path,"online") if online else os.path.join(path,"offline")
    return path

def form_plt_location_path(config):
    env_name = config["process_name"]
    location = config.get("plt_dir",f"plt")
    path = os.path.join(os.path.join(".", f"{env_name}"),f"{location}")
    return path

#----------------------------------------------------------------------
def parent_dir_and_name(file_path):
    return os.path.split(os.path.abspath(file_path))

def get_things_in_loc(in_path, just_files=True, endswith=None):
    if not os.path.exists(in_path):
        print(str(in_path) + " does not exists!")
        return
    re_list = []
    if not os.path.isdir(in_path):
        in_path = parent_dir_and_name(in_path)[0]

    for name in os.listdir(in_path):
        name_path = os.path.abspath(os.path.join(in_path, name))
        if os.path.isfile(name_path) and (endswith is None or (True in [name_path.endswith(ext) for ext in endswith])):
            re_list.append(name_path)
        elif not just_files:
            if os.path.isdir(name_path):
                re_list += get_things_in_loc(name_path, just_files=just_files, endswith=endswith)
    return re_list

#------------------------------------------------------------------------
import pandas as pd
class SeedData:
    """
    A dictionary that aims to average the evaluated mean_episode_return accross different random seed.
    Also controls where to resume the experiments from.
    """
    def __init__(self, save_path, seeds, resume_from={}):
        self.seed_data = pd.DataFrame({
            'algo_name': pd.Series([], dtype='str'),
            'test_reward': pd.Series([], dtype='float'),
            'seed': pd.Series([], dtype='int'),
        })
        self.save_path = save_path
        mkdir(save_path)
        self.load()
        # set experiment range
        self.resume_from = resume_from
        self.resume_check_passed = False
        self.seeds = seeds 

    def load(self):
        re_list = get_things_in_loc(self.save_path)
        if not re_list:
            print("Cannot find the a seed_data.csv at", self.save_path, "initializing a new one.")
            self.seed_data.to_csv(os.path.join(self.save_path, 'seed_data.csv'), index=False)
        else:
            self.seed_data = pd.read_csv(os.path.join(self.save_path, 'seed_data.csv'))
            print("Loaded the seed_data.csv at", self.save_path)
    
    def save(self):
        self.seed_data.to_csv(os.path.join(self.save_path, 'seed_data.csv'), index=False)

    def append(self, algo_name, test_reward, seed):
        self.seed_data.loc[len(self.seed_data)] = [algo_name, test_reward, seed]

    def setter(self, algo_name, test_reward, seed):
        # average over seed makes seed==-1
        # online makes dataset_percent==0.0
        self.append(algo_name, test_reward, seed)
        averaged_reward = self.seed_data.loc[(self.seed_data['algo_name'] == algo_name)]['test_reward'].mean()
        if seed == self.seeds[-1]: # append the average, seed now set to -1
            self.seed_data.loc[len(self.seed_data)] = [algo_name, averaged_reward, -1]
        self.save()
        return averaged_reward

    def resume_checker(self, current_positions):
        """
        current_positions has the same shape as self.resume_from
        return True if the current loop still need to be skipped.
        """
        if self.resume_check_passed is True: # checker has already passed.
            return True

        if not self.resume_from:
            self.resume_check_passed = True
        elif all([self.resume_from[condition] is None for condition in self.resume_from]):
            self.resume_check_passed = True
        else:
            self.resume_check_passed = all([self.resume_from[condition] == current_positions[condition] for condition in self.resume_from])
        return self.resume_check_passed
    
#-----------------------------------------------------------------------