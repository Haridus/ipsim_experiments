import d3rlpy
import os

#==============================================================================================================
def default_scorers_setup():
    return {
                'td_error': d3rlpy.metrics.scorer.td_error_scorer,
                'value_scale': d3rlpy.metrics.scorer.average_value_estimation_scorer,
                'discounted_sum_of_advantage_scorer': d3rlpy.metrics.scorer.discounted_sum_of_advantage_scorer,
                'value_estimation_std_scorer': d3rlpy.metrics.scorer.value_estimation_std_scorer,
                'initial_state_value_estimation_scorer': d3rlpy.metrics.scorer.initial_state_value_estimation_scorer,
                'continuous_action_diff_scorer': d3rlpy.metrics.scorer.continuous_action_diff_scorer,
            }

def setup_alg_CQL(config):
    return d3rlpy.algos.CQL(q_func_factory='qr', use_gpu=True, scaler = config['scaler'], action_scaler=config['action_scaler'], reward_scaler=config['reward_scaler']), default_scorers_setup()

def setup_alg_PLAS(config):
    return d3rlpy.algos.PLAS(q_func_factory='qr', use_gpu=True, scaler = config['scaler'], action_scaler=config['action_scaler'], reward_scaler=config['reward_scaler']), default_scorers_setup()

def setup_alg_PLASWithPerturbation(config):
    return d3rlpy.algos.PLASWithPerturbation(q_func_factory='qr', use_gpu=True, scaler = config['scaler'], action_scaler=config['action_scaler'], reward_scaler=config['reward_scaler']), default_scorers_setup()

def setup_alg_DDPG(config):
    return d3rlpy.algos.DDPG(use_gpu=True, scaler = config['scaler'], action_scaler=config['action_scaler'], reward_scaler=config['reward_scaler']), default_scorers_setup()

def setup_alg_BC(config):
    scorers = {'continuous_action_diff_scorer': d3rlpy.metrics.scorer.continuous_action_diff_scorer, }
    return d3rlpy.algos.BC(use_gpu=True, scaler = config['scaler'], action_scaler=config['action_scaler'], reward_scaler=config['reward_scaler']), scorers

def setup_alg_TD3(config):
    return d3rlpy.algos.TD3(use_gpu=True, scaler = config['scaler'], action_scaler=config['action_scaler'], reward_scaler=config['reward_scaler']), default_scorers_setup()

def setup_alg_BEAR(config):
    return d3rlpy.algos.BEAR(use_gpu=True, scaler = config['scaler'], action_scaler=config['action_scaler'], reward_scaler=config['reward_scaler']), default_scorers_setup()

def setup_alg_SAC(config):
    return d3rlpy.algos.SAC(use_gpu=True, scaler = config['scaler'], action_scaler=config['action_scaler'], reward_scaler=config['reward_scaler']), default_scorers_setup()

def setup_alg_BCQ(config):
    return d3rlpy.algos.BCQ(use_gpu=True, scaler = config['scaler'], action_scaler=config['action_scaler'], reward_scaler=config['reward_scaler']), default_scorers_setup()

def setup_alg_CRR(config):
    return d3rlpy.algos.CRR(use_gpu=True, scaler = config['scaler'], action_scaler=config['action_scaler'], reward_scaler=config['reward_scaler']), default_scorers_setup()

def setup_alg_AWR(config):
    scorers = {
                'td_error': d3rlpy.metrics.scorer.td_error_scorer,
                'value_scale': d3rlpy.metrics.scorer.average_value_estimation_scorer,
                'discounted_sum_of_advantage_scorer': d3rlpy.metrics.scorer.discounted_sum_of_advantage_scorer,
                'initial_state_value_estimation_scorer': d3rlpy.metrics.scorer.initial_state_value_estimation_scorer,
                'continuous_action_diff_scorer': d3rlpy.metrics.scorer.continuous_action_diff_scorer,
            }
    return d3rlpy.algos.AWR(use_gpu=True, scaler = config['scaler'], action_scaler=config['action_scaler'], reward_scaler=config['reward_scaler']), scorers

def setup_alg_AWAC(config):
    return d3rlpy.algos.AWAC(use_gpu=True, scaler = config['scaler'], action_scaler=config['action_scaler'], reward_scaler=config['reward_scaler'])

def setup_alg_COMBO(config):
    scorers={
        'observation_error': d3rlpy.metrics.scorer.dynamics_observation_prediction_error_scorer,
        'reward_error': d3rlpy.metrics.scorer.dynamics_reward_prediction_error_scorer,
        'variance': d3rlpy.metrics.scorer.dynamics_prediction_variance_scorer,
    }
    logdir = config['logdir']
    dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=1e-4, use_gpu=True, scaler = config['scaler'], action_scaler=config['action_scaler'], reward_scaler=config['reward_scaler'])
    dynamics.fit(config['feeded_episodes'],
        eval_episodes=config['eval_feeded_episodes'],
                n_epochs=config['DYNAMICS_N_EPOCHS'],
                logdir=logdir,
                scorers=scorers)
    return d3rlpy.algos.COMBO(dynamics=dynamics, use_gpu=True, scaler = config['scaler'], action_scaler=config['action_scaler'], reward_scaler=config['reward_scaler']), scorers

def setup_alg_MOPO(config):
    scorers={
        'observation_error': d3rlpy.metrics.scorer.dynamics_observation_prediction_error_scorer,
        'reward_error': d3rlpy.metrics.scorer.dynamics_reward_prediction_error_scorer,
        'variance': d3rlpy.metrics.scorer.dynamics_prediction_variance_scorer,
    }
    logdir = config['logdir']
    dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=1e-4, use_gpu=True, scaler = config['scaler'], action_scaler=config['action_scaler'], reward_scaler=config['reward_scaler'])
    dynamics.fit(config['feeded_episodes'],
        eval_episodes=config['eval_feeded_episodes'],
                n_epochs=config['DYNAMICS_N_EPOCHS'],
                logdir=logdir,
                scorers=scorers)
    return d3rlpy.algos.MOPO(dynamics=dynamics, use_gpu=True, scaler = config['scaler'], action_scaler=config['action_scaler'], reward_scaler=config['reward_scaler']), scorers

#======================================================
def get_rl_model(alg_name, logs_location):
    params = os.path.join(logs_location, f'{alg_name}/params.json')
    ckpt = os.path.join(logs_location, f'{alg_name}/best.pt')
    try:
        cls = getattr(d3rlpy.algos, alg_name)
        algorithm = cls.from_json(params)
        algorithm.load_model(ckpt)
        return algorithm
    except Exception as e:
        print(e)

class RLModel(object):
    def __init__(self, algo_name, logs_location):
        self.algorithm = get_rl_model(algo_name, logs_location)

    def predict(self, state):
        # for case of different input shape
        try:
            inp = self.algorithm.predict(state)  
            return inp[0] 
        except AssertionError:
            inp = self.algorithm.predict([state])
            return inp[0]
    
#=========================================================

