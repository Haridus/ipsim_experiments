import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../ipsim")) 

from mgym.env_wrapper import *

from ipsim import *

from scipy.integrate import solve_ivp

#================================================================================
def create_env_ECSTR_S0(config):
    MAX_OBSERVATIONS = [1.0, 100.0, 1.0]  
    MIN_OBSERVATIONS = [1e-08, 1e-08, 1e-08]
    MAX_ACTIONS = [35.0, 0.2] 
    MIN_ACTIONS = [15.0, 0.05]
    STEADY_OBSERVATIONS = [0.8778252, 51.34660837, 0.659]
    STEADY_ACTIONS = [26.85, 0.1]
    ERROR_REWARD = 0 #-1000.0

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
           
    env = EnvGym(  normalize=config.get("normalize", False)
                 , dense_reward=config.get("dense_reward", True)
                 , debug_mode= config.get("debug_mode", False)
                 , compute_diffs_on_reward = config.get("compute_diffs_on_reward", False)
                 , initial_state_deviation_ratio = config.get("initial_state_deviation_ratio", 0.3)
                 , sampling_time = config.get("sampling_time", 0.1)
                 , max_steps = config.get("initial_state_deviation_ratio", 100)
                 , action_dim = 2
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

def create_pid_conrol_ECSTR_S0(config):
    from controls.ECSTR_S0 import PID as ECSTR_S0_PID
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
def create_env_DistillationColumn(config):
    MAX_OBSERVATIONS = [1.0,]  
    MIN_OBSERVATIONS = [1e-08,]
    MAX_ACTIONS = [10.0,] 
    MIN_ACTIONS = [0.5,]
    STEADY_OBSERVATIONS = [0.80,]
    STEADY_ACTIONS = [2.5,]
    ERROR_REWARD = 0 #-1000.0

    def process_model_creator( dt = config.get("sampling_time", 0.1), initial_state = None ):
        from ipsim.models import DistillationColumn
        _observer    = ProcessModel.make_common_objerver( [ ("SensorXd", "xd"), ] )
    
        def npobserver(model, state):
            return np.array(_observer(model, state))
        
        _manipulator = ProcessModel.make_common_manipulator([ ("Reflux","ratio"), ])
        init_state = { "x": [ 0.935,0.900,0.862,0.821,0.779,0.738, \
                              0.698,0.661,0.628,0.599,0.574,0.553,0.535,0.521,    \
                              0.510,0.501,0.494,0.485,0.474,0.459,0.441,0.419,    \
                              0.392,0.360,0.324,0.284,0.243,0.201,0.161,0.125,    \
                              0.092,0.064] }

        solver = lambda f, ts, x, u, p: solve_ivp(f, ts, x, args = (u, p, ), method='LSODA')

        process =  DistillationColumn( solver=solver
                                     , observer=npobserver
                                     , manipulator=_manipulator
                                     , dt=1
                                     , init_state=init_state)
    

        #skip some steps to simulate random state
        u = [3,] 
        for _ in range(random.randint(0,300)):
            process.step(action = u)

        return process
           
    env = EnvGym(  normalize=config.get("normalize", False)
                 , dense_reward=config.get("dense_reward", True)
                 , debug_mode= config.get("debug_mode", False)
                 , compute_diffs_on_reward = config.get("compute_diffs_on_reward", False)
                 , initial_state_deviation_ratio = config.get("initial_state_deviation_ratio", 0.3)
                 , sampling_time = config.get("sampling_time", 1)
                 , max_steps = config.get("max_steps", 100)
                 , action_dim = 1
                 , observation_dim = 1
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

def create_pid_conrol_DistillationColumn(config):
    from controls.DistillationColumn import PIDController as DC_PID
    control =  DC_PID(0.8, Kp=1.25, gues=3, dt=1, min=0.1, max = 10)
    return control

#---------------------------------------------------------------------------------
def create_env_STEP(config):
    def process_model_creator( dt = config.get("sampling_time", 0.1), initial_state = None ):
        from ipsim.models import STEP

        def step_observer(model, state):
            f1 = state['Sensorf1']['f1'] #1
            f2 = state['Sensorf2']['f2'] #1
            F3 = state['SensorF3']['F3'] #3
            F4 = state['SensorF4']['F4'] #2
            Vl = state['SensorVl']['Vl'] #1
            P  = state['SensorP']['P']   #1
            X  = state['SensorX']['X']   #4

            return [ X[0], X[1], X[2], X[3]
                   , f1, f2, F3.F, F4.F
                   , F3.Comp[0], F3.Comp[1], F3.Comp[2]
                   , P, Vl ]
        
        def npobserver(model, state):
            return np.array(step_observer(model,state))
    
        manipulator = ProcessModel.make_common_manipulator([("ValvesControl","X"), ])
        solver = lambda f, ts, x, u, p: solve_ivp(f, ts, x, args = (u, p, ), method='LSODA')

        process = STEP(  solver = solver
                , observer = step_observer
                , manipulator = manipulator
                , init_state = initial_state)
        
        return process

    MAX_OBSERVATIONS = [ 1.0 #X1
                       , 1.0 #X2
                       , 1.0 #X3
                       , 1.0 #X4
                       , 330.46 #f1
                       , 22.46  #f2
                       , 350    #F3 
                       , 350    #F4
                       , 1.0    #aF3
                       , 1.0    #bF3
                       , 1.0    #cF3
                       , 3000   #P
                       , 1.0]   #Vl
    MIN_OBSERVATIONS = [ 1e-08 #X1
                       , 1e-08 #X2
                       , 1e-08 #X3
                       , 1e-08 #X4
                       , 1e-08 #f1
                       , 1e-08  #f2
                       , 1e-08    #F3 
                       , 1e-08    #F4
                       , 1e-08    #aF3
                       , 1e-08    #bF3
                       , 1e-08    #cF3
                       , 1e-08   #P
                       , 1e-08]   #Vl
    
    MAX_ACTIONS = [1.0, 1.0, 1.0, 1.0] 
    MIN_ACTIONS = [1e-08, 1e-08, 1e-08, 1e-08]
    STEADY_OBSERVATIONS = [0.8778252, 51.34660837, 0.659]
    STEADY_ACTIONS = [26.85, 0.1]
    ERROR_REWARD = -10000.0

    env = EnvGym(  normalize=config.get("normalize", False)
                 , dense_reward=config.get("dense_reward", True)
                 , debug_mode= config.get("debug_mode", False)
                 , compute_diffs_on_reward = config.get("compute_diffs_on_reward", False)
                 , initial_state_deviation_ratio = config.get("initial_state_deviation_ratio", 0.3)
                 , sampling_time = config.get("sampling_time", 0.1)
                 , max_steps = config.get("initial_state_deviation_ratio", 100)
                 , action_dim = 2
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