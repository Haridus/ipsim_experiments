import pickle
from .utils import *
from tqdm import tqdm
import numpy as np

#===========================================================================
def generate_dataset(env, control, config, num_of_initial_state):
    normalize = config.get("normalize",False)
    initial_states = []
    for _ in tqdm(range(num_of_initial_state)):
        init = env.sample_initial_state()
        initial_states.append(init)
    dataset = env.generate_dataset_with_algorithm(control, normalize=normalize, num_episodes=len(initial_states), 
    initial_states=initial_states, format='d4rl')

    return dataset

def form_dataset_location_path(config):
    env_name = config["process_name"]
    dataset_location = config.get("dataset_location",f"datasets")
    dataset_path = os.path.join(os.path.join(".", f"{env_name}"),f"{dataset_location}")
    return dataset_path

def save_dataset(dataset, config, dataset_name):
    normalize = config.get("normalize",False)
    dataset_path = form_dataset_location_path(config)
    mkdir(dataset_path)
    
    dataset_loc = os.path.join(dataset_path, f'{dataset_name}_normalize={normalize}.pkl')
    with open(dataset_loc, 'wb') as fp:
        pickle.dump(dataset, fp)
    print(f"saved dataset {dataset_loc}")