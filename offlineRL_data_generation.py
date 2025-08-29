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
    parser.add_argument('-p','--process', type = str, default = 'ECSTR_S0', help = 'Process model name')
    parser.add_argument('-w','--work_dir', type = str, default=os.path.dirname(__file__), help = 'Working directory')
    args = parser.parse_args()
    
    os. chdir(args.work_dir)

    config = load_config_yaml(args.work_dir, args.process)
    env = EnvFactory.create(config=config)
    control = ControlFactory.create(config=config)
    dataset = generate_dataset(env, control, config)
    
    np.set_printoptions(threshold=sys.maxsize)
    print(dataset["rewards"])

    save_dataset(dataset, config)
