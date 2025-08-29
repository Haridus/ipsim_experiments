from utils.utils import *
from utils.constructors import *
from utils.data_generation import *

EnvFactory.constructors['ECSTR_S0'] = create_env_ECSTR_S0
ControlFactory.constructors['ECSTR_S0'] = create_pid_conrol_ECSTR_S0

EnvFactory.constructors['ReactorEnv'] = create_env_ReactorEnv
ControlFactory.constructors['ReactorEnv'] = create_pid_control_ReactorEnv

class MetricsCalculator:
    def __init__(self, setpoints, dt = 1):
        self.setpoints = setpoints
        self.dt = dt

        self.error     = [0 for x in range(len(self.setpoints))]

        self.sums_ISF  = [0 for x in range(len(self.setpoints))]
        self.sums_IAE  = [0 for x in range(len(self.setpoints))]
        self.sums_ITAE = [0 for x in range(len(self.setpoints))]
        self.sums_ITSH = [0 for x in range(len(self.setpoints))]
    
    def update(state):
        for _ in range(len(state)):
            pass

if __name__ == "__main__":
    pass

"""
def calculate_control_quality_metrics(e, dt):
    sum_ISF  = 0
    sum_IAE  = 0
    sum_ITAE = 0
    sum_ITSH = 0
    t = 0
    for _ in range(0, len(e)-1):
        t += dt
        sum_ISF  += e[_]**2+e[_+1]**2
        sum_IAE  += abs(e[_])+abs(e[_+1])
        sum_ITAE += (abs(e[_])+abs(e[_+1]))*t
        sum_ITSH += (e[_]**2+e[_+1]**2)*t

    return { "ISF":sum_ISF*dt/2
           , "IAE":sum_IAE*dt/2
           , "ITAE":sum_ITAE*dt/2
           , "ITSH":sum_ITSH*dt/2}
"""