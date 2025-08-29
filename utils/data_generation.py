import pickle
from .utils import *
from tqdm import tqdm
import numpy as np

#===========================================================================
def generate_dataset(env, control, config):
    normalize = config.get("normalize",False)
    num_of_initial_state = config.get("num_of_initial_state",1000)
    initial_states = []
    for _ in tqdm(range(num_of_initial_state)):
        init = env.sample_initial_state()
        initial_states.append(init)
    dataset = env.generate_dataset_with_algorithm(control, normalize=normalize, num_episodes=len(initial_states), 
    initial_states=initial_states, format='d4rl')

    return dataset

def save_dataset(dataset, config):
    env_name = config["model_name"]
    normalize = config.get("normalize",False)
    dataset_location = config.get("dataset_location",f"./{env_name}")
    num_of_initial_state = config.get("num_of_initial_state",1000)
    mkdir(dataset_location)
    dataset_loc = os.path.join(dataset_location, f'{num_of_initial_state}_normalize={normalize}.pkl')
    #print("terminals", np.where(dataset["terminals"] == True))
    #print("timeouts", np.where(dataset["timeouts"] == True))
    with open(dataset_loc, 'wb') as fp:
        pickle.dump(dataset, fp)
    print(f"saved dataset {dataset_loc}")