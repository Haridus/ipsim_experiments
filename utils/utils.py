import yaml

#=====================================================================
class EnvFactory:
    constructors = { "SimpleCSTR": create_env_SCSTR
                   , "ReactorEnv": create_env_ReactorEnv
                   }
    
    @staticmethod
    def create(config):
        env_name = config['model_name']
        if env_name in EnvFactory.constructors:
            return EnvFactory.constructors[env_name](config=config)
        raise Exception("Unknown env")

class ControlFactory:
    constructors = { "SimpleCSTR": create_pid_conrol_SCSTR
                   , "ReactorEnv": create_pid_control_ReactorEnv
                   }

    @staticmethod
    def create(config):
        env_name = config['model_name']
        if env_name in ControlFactory.constructors:
            return ControlFactory.constructors[env_name](config=config)
        raise Exception("Unknown env")

#----------------------------------------------------------------------
def load_config_yaml(work_dir, model_name):
    config_path = f'{work_dir}/{model_name}.yaml'
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)
        config['work_dir'] = work_dir
        config['model_name'] = model_name
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

#----------------------------------------------------------------------