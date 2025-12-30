from ipsim import *
from ipsim.models import ECSTR_A0_Node, ECSTR_A0

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#===============================================================================
class PID:
    def __init__(self, setpoint, Kp
                , *, Ki = 0, Kd = 0, dt = 0
                , min_value = 0
                , max_value = 0 ):
        self.setpoint = setpoint
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.min_value = min_value
        self.max_value = max_value
        self.integral = 0
        self.prev_error = 0
        self.error = 0

    def __call__(self, meas, manipulated_variable_value = 0):
        self.error = self.setpoint - meas
        P = self.Kp*self.error

        I = 0
        if self.Ki >= 0 and self.dt > 0:
            self.integral += self.error*self.dt
            I = self.Ki*self.integral

        D = 0
        if self.Kd >= 0 and self.dt > 0:
            derivative = (self.error - self.prev_error)/self.dt
            D = self.Kd*derivative

        self.prev_error = self.error
        value = np.clip(P+I+D + manipulated_variable_value, a_min=self.min_value, a_max= self.max_value)

        return value

#-----------------------------------------------------------------------------------    
def run(process_model, controller = None, *, iteration = 1000):
    cAs  = []
    cBs  = []
    Ts   = []
    actions = []
    coolant_node = process_model.nodes()["Coolant"]
    action = [coolant_node.value("T"), ]
    for i in range(iteration):
        state = process_model.step(action = action)
        cAs.append(state[0])
        cBs.append(state[1])
        Ts.append(state[2])
        
        if controller is not None:
            newTc = controller(state[2], coolant_node.value("T") )
            action[0] = newTc 
            actions.append(newTc)
        else:
            actions.append(coolant_node.value("T"))

    return {"cAs":cAs, "cBs":cBs, "Ts":Ts, "Actions":actions}

def show_metrics(data):
    plt.figure(figsize=(12,3))
    plt.subplot(1, 4, 1)
    plt.plot(data["cAs"])
    plt.title('Concentration [mol/L]')
    plt.subplot(1, 4, 2)
    plt.plot(data["cBs"])
    plt.title('Concentration [mol/L]')
    plt.subplot(1, 4, 3)
    plt.plot(data["Ts"])
    plt.title('Temperature [K]')
    plt.subplot(1, 4, 4)
    plt.plot(data["Actions"])
    plt.title('Temperature [K]')

if __name__ == "__main__":
    observer    = ProcessModel.make_common_objerver( [ ("SensorA", "cA")
                                                   , ("SensorB", "cB")
                                                   , ("SensorT", "T") ])
    manipulator = ProcessModel.make_common_manipulator([("Coolant","T"), ])

    solver = lambda f, ts, x, u, p: solve_ivp(f, ts, x, args = (u, p, ), method='LSODA')
    init_state = {"cA":0.5, "cB":0, "T":350}
    scstr = ECSTR_A0( solver = solver
                    , observer = observer
                    , manipulator = manipulator
                    , init_state = init_state)
    
    pid = PID(setpoint=385, Kp=0.03125, Ki=2.842170943040401e-16, Kd = 0.0075, dt=scstr.dt(), min_value=298, max_value=395)

    data = run(scstr, controller=pid)
    show_metrics(data)
    plt.show()
    