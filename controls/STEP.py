import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../ipsim")) 

from ipsim import *
from ipsim.models import *

import numpy as np
import random as rnd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from copy import deepcopy

#===============================================================================
class STEPMultiloopPControl:
    def __init__(self, F4_setpoint, P_setpoint, cA3_setpoint, dt = 0.1):
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
      
    def __call__(self, meas, mvs = None, *args, **kwds):
        #return [ X[0], X[1], X[2], X[3]
        #               , f1, f2, F3.F, F4.F
        #               , F3.Comp[0], F3.Comp[1], F3.Comp[2]
        #               , P, Vl ]

        errnPC = (2900 - meas[11])
        self.F4sp_adj = min(self.F4sp_adj+self.KcPC*( (errnPC-self.errn1PC)+(self.dt*errnPC)/self.TauiPC ), 0)
        self.errn1PC=errnPC

        F4_setpoint = self.F4_setpoint + self.F4sp_adj

        F4errn  = F4_setpoint  - meas[7]
        Perrn   = self.P_setpoint   - meas[11]
        cA3errn = (self.cA3_setpoint - meas[10])*100 # since Kcs for address valves positions from 0 to 100, but here they witheen range 0 to 1
        
        F4delu  = self.Kcs[0]*( (F4errn  - self.F4errn1)+(self.dt*F4errn)/self.tauis[0] )
        Pdelu   = self.Kcs[1]*( (Perrn   - self.Perrn1)+(self.dt*Perrn)  /self.tauis[1] )
        cA3delu = self.Kcs[2]*( (cA3errn - self.cA3errn1)+(self.dt*cA3errn)/self.tauis[2] )
        
        X_target = [ np.clip(mvs[0] + F4delu/100, a_min= 0, a_max=1) #since Kcs and tauis is specified for X form 0 to 100% and here X from 0 to 1dd
                   , np.clip(mvs[1] + cA3delu/100, a_min= 0, a_max=1)
                   , np.clip(mvs[2] + Pdelu/100, a_min= 0, a_max=1)
        ]
        X_target.append(mvs[3])
        #print(f"X:{X_target}")
        
        self.F4errn1  = F4errn
        self.Perrn1   = Perrn
        self.cA3errn1 = cA3errn
        
        return X_target

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
    x = np.empty((iterations,13))

    parameters_metadata = (
          {"parameter":"x1", "range":(0,1), "units": "", "title":"valve 1 pos"}
         ,{"parameter":"x2", "range":(0,1), "units": "", "title":"valve 2 pos"}
         ,{"parameter":"x3", "range":(0,1), "units": "", "title":"valve 3 pos"}
         ,{"parameter":"x4", "range":(0,1), "units": "", "title":"valve 4 pos"}
         ,{"parameter":"F1", "range":(0,300), "units": "kmol", "title":"Flow 1"}
         ,{"parameter":"F2", "range":(0,30), "units": "kmol", "title":"Flow 2"}
         ,{"parameter":"F3", "range":(0,100), "units": "kmol", "title":"Flow 3"}
         ,{"parameter":"F4", "range":(0,200), "units": "kmol", "title":"Flow 4"}
         ,{"parameter":"cA", "range":(0,1), "units": "", "title":"mol. frac A"}
         ,{"parameter":"cB", "range":(0,1), "units": "", "title":"mol. frac B"}
         ,{"parameter":"cC", "range":(0,1), "units": "", "title":"mol. frac C"}
         ,{"parameter":"P", "range":(2500,3000), "units": "kPa", "title":"Pressure"}
         ,{"parameter":"vl",  "units": "", "range":(0,1), "title":"Liq. inv. fill rate"  }
         )

    valves_positions = process_model.nodes()["ValvesControl"].value("X")
    action = [valves_positions,]
    for _ in range(iterations):
        if controller is not None:
            state = process_model.step(action = action)
            valves_positions = controller(state, valves_positions)
            action[0] = valves_positions
        else:
            state = process_model.step()

        x[_] = state

    return x, parameters_metadata

#===================================================================================
if __name__ == "__main__":
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
    
    manipulator = ProcessModel.make_common_manipulator([("ValvesControl","X"), ])
    solver = lambda f, ts, x, u, p: solve_ivp(f, ts, x, args = (u, p, ), method='LSODA')

    step = STEP(  solver = solver
                , observer = step_observer
                , manipulator = manipulator)
    
    multiloop_control = None
    multiloop_control = STEPMultiloopPControl(F4_setpoint=100, P_setpoint=2700, cA3_setpoint=0.38, dt=step.dt())
    #F_4 = 130, P= 2850, yA3 = 0.63
    #multiloop_control = STEPMultiloopPControl(F4_setpoint=130.00, P_setpoint=2850.0, cA3_setpoint=0.63, dt=step.dt())
    def test_Control(self, meas, mvs = None, *args, **kwds):
        return [meas[0], meas[1], 1,meas[3], ]
        #return [meas[0], 0.4, meas[2],meas[3], ]
        #return [0.76, meas[1], meas[2],meas[3], ]
        #return [meas[0], 0, meas[2],meas[3], ]
    multiloop_control = test_Control
    
    data, metadata = run(step, controller=multiloop_control)
    show(data, metadata=metadata)
    plt.show()
    