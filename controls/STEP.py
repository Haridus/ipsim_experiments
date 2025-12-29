import numpy as np
import os
import sys

from ipsim import *
from ipsim.models import *

import numpy as np
import random as rnd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from copy import deepcopy

#===============================================================================
class STEPMultiloopPControl:
    def __init__(self, F4_setpoint, P_setpoint, cA3_setpoint, dt = 0.1, process = None):
        self.Kcs   = [0.1, -0.25, 2.0]
        self.tauis = [1.0, 1.5, 3.0] 
        self.errn1PC = 0
        self.KcPC       = 0.7
        self.TauiPC     = 3
        self.F4sp_adj   = 0
        self.F4_setpoint = F4_setpoint
        self.P_setpoint  = P_setpoint
        self.cA3_setpoint = cA3_setpoint 
        self.F4errn1  = 0
        self.Perrn1   = 0
        self.cA3errn1 = 0
        self.dt = dt
        self.proress = process
      
    def __call__(self, meas, mvs = None, *args, **kwds):
        errnPC = (2900 - meas[2])
        self.F4sp_adj = min(self.F4sp_adj+self.KcPC*( (errnPC-self.errn1PC)+(self.dt*errnPC)/self.TauiPC ), 0)
        self.errn1PC=errnPC

        F4_setpoint = self.F4_setpoint + self.F4sp_adj
        F4errn  = F4_setpoint  - meas[1]
        Perrn   = self.P_setpoint   - meas[2]
        cA3errn = (self.cA3_setpoint - meas[0])*100 # since Kcs for address valves positions from 0 to 100, but here they witheen range 0 to 1
        
        F4delu  = self.Kcs[0]*( (F4errn  - self.F4errn1)  + (self.dt*F4errn)/self.tauis[0] )
        Pdelu   = self.Kcs[1]*( (Perrn   - self.Perrn1)   + (self.dt*Perrn)  /self.tauis[1] )
        cA3delu = self.Kcs[2]*( (cA3errn - self.cA3errn1) + (self.dt*cA3errn)/self.tauis[2] )
        
        X = self.proress.process_model.nodes()['SensorX'].value("X")

        X_target = [ np.clip(X[0] + F4delu/100, a_min= 0, a_max=1) #since Kcs and tauis is specified for X form 0 to 100% and here X from 0 to 1dd
                   , np.clip(X[1] + cA3delu/100, a_min= 0, a_max=1)
                   , np.clip(X[2] + Pdelu/100, a_min= 0, a_max=1)
        ]

        self.F4errn1  = F4errn
        self.Perrn1   = Perrn
        self.cA3errn1 = cA3errn
        
        return X_target
    
    def predict(self, measured_value, *args, **kwds) :
        nv = self.__call__(measured_value, None, *args, **kwds)
        return nv
    
#===============================================================================
def show(x, metadata, title = "", *, cols=4, dt=0.1):
    fig = plt.figure(figsize=(6,6),)
    fig.suptitle(title)
    paramsCount = len(metadata)
    rows = int(paramsCount / cols) + (1 if paramsCount % cols > 0 else 0 )
    
    for r in range(0, rows):
        for c in range(0,cols):
            index = r*cols + c
            if index < paramsCount:
                plt.subplot(rows, cols, index+1)
                if metadata[index]["range"] is not None:
                    plt.ylim(metadata[index]["range"][0], metadata[index]["range"][1])
                plt.plot([i*dt for i in range(len(x))] ,x[:,index])
                plt.title(metadata[index]["title"])
    
    plt.tight_layout()
    plt.show()

#==============================================================================
def run(process_model, *, controller = None, time_target=30):
    dt = process_model.dt()
    iterations = int( (time_target)/dt)
    x = np.empty((iterations,3))

    parameters_metadata = (
          {"parameter":"cA", "range":(0,1), "units": "", "title":"mol. frac A"}
         ,{"parameter":"F4", "range":(80,140), "units": "kmol", "title":"Flow 4"}
         ,{"parameter":"P", "range":(2600,3000), "units": "kPa", "title":"Pressure"}
         )
    
    state  = process_model.step()
    for _ in range(iterations):
        if controller is not None:
            action = controller(state)
            state = process_model.step(action = action)
        else:
            state = process_model.step()
        x[_] = state
    return x, parameters_metadata

#===================================================================================
config = {"sampling_time": 0.1}

def process_model_creator( dt = config.get("sampling_time", 0.1), initial_state = None ):
        from ipsim.models import STEP

        def step_observer(model, state):
            F3 = state['SensorF3']['F3'] #3
            F4 = state['SensorF4']['F4'] #2
            P  = state['SensorP']['P']   #1
            
            return np.array([F3.Comp[STEPFlow.A], F4.F, P, ])
        observer = step_observer

        def step_manipulator(model, action):
            uX = model.nodes()["ValvesControl"].value("X")
            uX[0] = action[0]
            uX[1] = action[1]
            uX[2] = action[2]

        manipulator = step_manipulator

        solver = lambda f, ts, x, u, p: solve_ivp(f, ts, x, args = (u, p, ), method='LSODA')
        process =  STEP(solver=solver, observer=observer, manipulator=manipulator, dt=0.1)
        return process

if __name__ == "__main__":
    step = process_model_creator(dt=0.1)
    for _ in range(100):
        step.step()
    print(step.step())

    multiloop_control = None
    #multiloop_control = STEPMultiloopPControl(F4_setpoint=99.99991893665879, P_setpoint=2700.1477279941, cA3_setpoint=0.4699865552748352, dt=step.dt())
    #F_4 = 130, P= 2850, yA3 = 0.63

    random_action = [rnd.randrange(1,100)/100.0, rnd.randrange(1,100)/100.0, rnd.randrange(1,100)/100.0]
    step.step(action=random_action)
    print(random_action)

    multiloop_control = STEPMultiloopPControl(F4_setpoint=130.00, P_setpoint=2850.0, cA3_setpoint=0.63, dt=step.dt(), process=step)
    data, metadata = run(step, controller=multiloop_control, time_target=30)
    show(data, metadata=metadata)
    plt.show()