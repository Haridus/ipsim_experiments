import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../ipsim")) 

from ipsim import *

from ..gym.env_wrapper import *

from scipy.integrate import solve_ivp

#================================================================================
def create_env_SCSTR(config):
    MAX_OBSERVATIONS = [1.0, 100.0, 1.0]  
    MIN_OBSERVATIONS = [1e-08, 1e-08, 1e-08]
    MAX_ACTIONS = [35.0, 0.2] 
    MIN_ACTIONS = [15.0, 0.05]
    STEADY_OBSERVATIONS = [0.8778252, 51.34660837, 0.659]
    STEADY_ACTIONS = [26.85, 0.1]
    ERROR_REWARD = -1000.0

    def process_model_creator( dt = config.get("sampling_time", 0.1), initial_state = None ):
        from ipsim.models import ECSTR_S0

        _observer    = ProcessModel.make_common_objerver( [ ("SensorA", "cA")
                                                        , ("SensorT", "T")
                                                        , ("SensorH", "h") ])
        def npobserver(model, state):
            return np.array(_observer(model,state))
    
        _manipulator = ProcessModel.make_common_manipulator([("Coolant","T"), ("OutFlowControl", "q"), ])
        init_state = { "cA": initial_state[0]
                     , "T" : initial_state[1]
                     , "h" : initial_state[2]}

        solver = lambda f, ts, x, u, p: solve_ivp(f, ts, x, args = (u, p, ), method='LSODA')

        process =  ECSTR_S0(solver=solver, observer=npobserver, manipulator=_manipulator, dt=0.1, init_state=init_state)
        return process
           
    env = EnvGym(  normalize=config.get("normalize", True)
                 , dense_reward=config.get("dense_reward", True)
                 , debug_mode= config.get("debug_mode", False)
                 , compute_diffs_on_reward = config.get("compute_diffs_on_reward", False)
                 , initial_state_deviation_ratio = config.get("initial_state_deviation_ratio", 0.3)
                 , sampling_time = config.get("sampling_time", 0.1)
                 , max_steps = config.get("initial_state_deviation_ratio", 100)
                 , action_dim = 1
                 , observation_dim = 3
                 , max_observations=MAX_OBSERVATIONS
                 , min_observations=MIN_OBSERVATIONS
                 , max_actions=MAX_ACTIONS
                 , min_actions=MIN_ACTIONS
                 , steady_observation = STEADY_OBSERVATIONS
                 , steady_action      = STEADY_ACTIONS
                 , error_reward=ERROR_REWARD
                 , np_dtype=np.float32
                 , process_model_constructor=process_model_creator)
    return env

def create_pid_conrol_SCSTR(config):
    from ..controls.ECSTR_S0 import PID as ECSTR_S0_PID
    control =  ECSTR_S0_PID(Kis=[100.0, 0.5], steady_state=[0.8778252, 0.659], steady_action=[26.85, 0.1], min_action=[15.0, 0.05], max_action=[35.0, 0.2])
    return control

#----------------------------------------------------------------------------------
def create_env_ReactorEnv(config):
    from smpl.envs.reactorenv import ReactorEnvGym
    env = ReactorEnvGym( normalize=config["normalize"]
                       , dense_reward=config["dense_reward"]
                       , compute_diffs_on_reward=config["compute_diffs_on_reward"])
    return env

def create_pid_control_ReactorEnv(config):
    from smpl.envs.reactorenv import ReactorPID
    return ReactorPID(Kis=[100.0, 0.5], steady_state=[0.8778252, 0.659], steady_action=[26.85, 0.1], min_action=[15.0, 0.05], max_action=[35.0, 0.2])

#---------------------------------------------------------------------------------