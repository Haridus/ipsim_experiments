#================================================================
from utils.utils import *
from utils.constructors import *
from utils.data_generation import *

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
    args = parser.parse_args()
    
    os. chdir(args.work_dir)

    config = load_config_yaml(args.work_dir, args.process)
    env = EnvFactory.create(config=config)
    control = ControlFactory.create(config=config)

    eval_num_of_initial_state = config.get('eval_num_of_initial_state',1000)
    training_num_of_initial_state = config.get('training_num_of_initial_state',1000)
    eval_dataset = generate_dataset(env, control, config, eval_num_of_initial_state )
    training_dataset = generate_dataset(env, control, config, training_num_of_initial_state )
    save_dataset(training_dataset, config, f'train_{training_num_of_initial_state}')
    save_dataset(eval_dataset, config, f'test_{eval_num_of_initial_state}')
