import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../ipsim")) 

from ipsim import *
from ipsim.models import ECSTR_S0_Node, ECSTR_S0

from ipsim.ipsim import *
from numpy import exp, seterr, pi
import numpy as np
seterr(all='ignore')

from scipy.integrate import solve_ivp

import pandas as pd
import matplotlib.pyplot as plt

#=================================================================================
class PID:
    def __init__(self, Kis, steady_state=[0.8778252, 0.659], steady_action=[26.85, 0.1], min_action=[15.0, 0.05],
                 max_action=[35.0, 0.2]):
        self.Kis = Kis
        self.steady_state = steady_state
        self.steady_action = steady_action
        self.len_c = len(steady_action)
        self.min_action = min_action
        self.max_action = max_action

    def predict(self, state):
        state = [state[0], state[2]]
        action = []
        for i in range(self.len_c):
            a = self.Kis[i] * (state[i] - self.steady_state[i]) + self.steady_action[i]
            action.append(np.clip(a, self.min_action[i], self.max_action[i]))
        return np.array(action)
    
def show_metrics(data):
    plt.figure(figsize=(12,3))
    plt.subplot(1, 5, 1)
    plt.plot(data["cs"])
    plt.subplot(1, 5, 2)
    plt.plot(data["Ts"])
    plt.subplot(1, 5, 3)
    plt.plot(data["hs"])
    plt.subplot(1, 5, 4)
    plt.plot(data["Tcs"])
    plt.subplot(1, 5, 5)
    plt.plot(data["qs"])

if __name__ == "__main__":
    _observer    = ProcessModel.make_common_objerver( [ ("SensorA", "cA")
                                                        , ("SensorT", "T")
                                                        , ("SensorH", "h") ])
    
    def npobserver(model, state):
            return np.array(_observer(model,state))
    
    _manipulator = ProcessModel.make_common_manipulator([("Coolant","T"), ("OutFlowControl", "q"), ])

    state = [0.1, 1, 0.1]
    init_state = { "cA": state[0]
                 , "T" : state[1]
                 , "h" : state[2]}

    solver = lambda f, ts, x, u, p: solve_ivp(f, ts, x, args = (u, p, ), method='LSODA')

    process =  ECSTR_S0(solver=solver, observer=npobserver, manipulator=_manipulator, dt=0.1, init_state=init_state)
    control =  PID(Kis=[100.0, 0.5], steady_state=[0.8778252, 0.659], steady_action=[26.85, 0.1], min_action=[15.0, 0.05], max_action=[35.0, 0.2])

    cs  = []  # c_A
    Ts  = []  # T
    hs  = []  # h
    Tcs = []  # Tc
    qs =  []  # q_out
    for _ in range(1000):
        cs.append(state[0])
        Ts.append(state[1])
        hs.append(state[2])
        u = control.predict(state)
        
        Tcs.append(u[0])
        qs.append(u[1])
        state = process.step(action=u)

    data = {"cs":cs, "Ts":Ts, "hs":hs, "Tcs": Tcs, "qs":qs}
    show_metrics(data)
    plt.show()
    
    df = pd.DataFrame()
    df["cs"] = data["cs"]
    df["Ts"] = data["Ts"]
    df["hs"] = data["hs"]
    df["Tcs"] = data["Tcs"]
    df["qs"] = data["qs"]
    print(df)