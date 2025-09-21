from utils.utils import *
from utils.constructors import *
from utils.data_generation import *

EnvFactory.constructors['ECSTR_S0'] = create_env_ECSTR_S0
ControlFactory.constructors['ECSTR_S0'] = create_pid_conrol_ECSTR_S0

EnvFactory.constructors['ReactorEnv'] = create_env_ReactorEnv
ControlFactory.constructors['ReactorEnv'] = create_pid_control_ReactorEnv

from copy import deepcopy
#=======================================================================
class MetricsCalculator:
    def __init__(self, setpoints, dt = 1):
        self.setpoints = setpoints
        self.dt = dt

        self.t = 0
        self.error     = [0 for x in range(len(self.setpoints))]
        self.error2     = [0 for x in range(len(self.setpoints))]
        self.sums_ISF  = [0 for x in range(len(self.setpoints))]
        self.sums_IAE  = [0 for x in range(len(self.setpoints))]
        self.sums_ITAE = [0 for x in range(len(self.setpoints))]
        self.sums_ITSH = [0 for x in range(len(self.setpoints))]
    
    def update(self, state):
        self.t += self.dt
        for _ in range(min(len(self.setpoints), len(state))):
            e = self.setpoints[_] - state[_]
            self.error[_] += e
            self.sums_ISF[_]  += e**2
            self.sums_IAE[_]  += abs(e)
            self.sums_ITAE[_] += abs(e)*self.t
            self.sums_ITSH[_] += (e**2)*self.t
            print(f"{self.t}: {e} : {self.setpoints} {state} {self.sums_ISF} ")

    def ISF(self):
        return [sum*self.dt/2 for sum in self.sums_ISF]
    
    def IAE(self):
        return [sum*self.dt/2 for sum in self.sums_IAE]
    
    def ITAE(self):
        return [sum*self.dt/2 for sum in self.sums_ITAE]
    
    def ITSH(self):
        return [sum*self.dt/2 for sum in self.sums_ITSH]
    


def show_metrics(data1, data2):
    d1_color_mark = "r"
    d2_color_mark = "b"
    
    plt.figure(figsize=(12,3))
    plt.subplot(1, 5, 1)
    plt.plot(data1["Cs"], color=d1_color_mark)
    plt.plot(data2["Cs"], color=d2_color_mark)
    
    plt.subplot(1, 5, 2)
    plt.plot(data1["Ts"], color=d1_color_mark)
    plt.plot(data2["Ts"], color=d2_color_mark)
    
    plt.subplot(1, 5, 3)
    plt.plot(data1["hs"], color=d1_color_mark)
    plt.plot(data2["hs"], color=d2_color_mark)
    
    plt.subplot(1, 5, 4)
    plt.plot(data1["Tcs"], color=d1_color_mark)
    plt.plot(data2["Tcs"], color=d2_color_mark)
    
    plt.subplot(1, 5, 5)
    plt.plot(data1["qs"], color=d1_color_mark)
    plt.plot(data2["qs"], color=d2_color_mark)
    
if __name__ == "__main__":
    os. chdir(os.path.dirname(__file__))

    init_state = [0.1, 1, 0.1]
    setpoints  = [0.8778252, 0.659]
    ref_state = deepcopy(init_state)

    ref_config = load_config_yaml(".", "ReactorEnv")
    ref_env = EnvFactory.create(ref_config)
    ref_env.reset()
    ref_process = ref_env.reactor
    ref_process_control = ControlFactory.create(ref_config)
    dt = ref_env.sampling_time

    ref_process_metrics_calculator = MetricsCalculator(setpoints=setpoints,dt = dt)
    ref_process_data = {"Cs":[], "Ts":[], "hs":[], "Tcs": [], "qs":[]}
    
    config = load_config_yaml(".", "ECSTR_S0")
    config['model_name'] = 'ECSTR_S0'
    env = EnvFactory.create(config)
    state = deepcopy(init_state)
    process = env.process_model_constructor(dt=dt, initial_state = init_state)
    process_control = ControlFactory.create(config)

    process_metrics_calculator = MetricsCalculator(setpoints=setpoints,dt = dt)
    process_data = {"Cs":[], "Ts":[], "hs":[], "Tcs": [], "qs":[]}

    iterations = 100
    for _ in range(iterations):
        ref_u     = ref_process_control.predict(ref_state)
        ref_state = ref_process.step(ref_state, ref_u)
        ref_process_metrics_calculator.update((ref_state[0], ref_state[2], ))

        ref_process_data["Cs"].append(ref_state[0])
        ref_process_data["Ts"].append(ref_state[1])
        ref_process_data["hs"].append(ref_state[2])
        ref_process_data["Tcs"].append(ref_u[0])
        ref_process_data["qs"].append(ref_u[1])

        u = process_control.predict(state)
        state = process.step(action = u)
        process_metrics_calculator.update((state[0], state[2], ))
        process_data["Cs"].append(state[0])
        process_data["Ts"].append(state[1])
        process_data["hs"].append(state[2])
        process_data["Tcs"].append(u[0])
        process_data["qs"].append(u[1])

    show_metrics(ref_process_data, process_data)
    plt.show()

    print(f"ref process metrics: {ref_process_metrics_calculator.ISF()} ; {ref_process_metrics_calculator.IAE()}; {ref_process_metrics_calculator.ITAE()}; {ref_process_metrics_calculator.ITSH()}; ")
    print(f"process metrics: {process_metrics_calculator.ISF()}; {process_metrics_calculator.IAE()}; {process_metrics_calculator.ITAE()}; {process_metrics_calculator.ITSH()};")

    print(f"{ref_process_metrics_calculator.error} {ref_process_metrics_calculator.sums_ISF} {ref_process_metrics_calculator.sums_IAE} {ref_process_metrics_calculator.sums_ITAE} {ref_process_metrics_calculator.sums_ITSH}")

