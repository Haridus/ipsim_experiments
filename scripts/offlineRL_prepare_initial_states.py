#================================================================
from utils.utils import *
from utils.constructors import *
from utils.data_generation import *

import os
import argparse

EnvFactory.constructors['ECSTR_S0'] = create_env_ECSTR_S0
ControlFactory.constructors['ECSTR_S0'] = create_pid_conrol_ECSTR_S0

EnvFactory.constructors['ReactorEnv'] = create_env_ReactorEnv
ControlFactory.constructors['ReactorEnv'] = create_pid_control_ReactorEnv

#================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--process', type = str, default = 'ReactorEnv', help = 'Process model name')
    parser.add_argument('-w','--work_dir', type = str, default=os.path.dirname(__file__), help = 'Working directory')
    parser.add_argument('-v','--val_per_state', type = str, default=10, help = 'values per state')
    args = parser.parse_args()
    
    os. chdir(args.work_dir)

    config = load_config_yaml(args.work_dir, args.process)
    datasets_path = form_dataset_location_path(config)
    mkdir(datasets_path)

    initial_states_location = os.path.join(datasets_path,f'initial_states_step_{args.val_per_state}.npy')
    
    env = EnvFactory.create(config=config)
    env.evenly_spread_initial_states(args.val_per_state, dump_location=initial_states_location)
    