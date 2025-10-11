import numpy as np

#======================================================================
class MyPID:

    def __init__(self, setpoint, Kp, *, Ki = None, Kd = None, dt = 1):
        self.setpoint = setpoint
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0
        self.prev_error = 0
        self.error = 0

    def __call__(self, meas):
        self.error = self.setpoint - meas[0]
        P = self.Kp*self.error

        I = 0
        if self.Ki is not None:
            self.integral += self.error*self.dt
            I = self.Ki*self.integral

        D = 0
        if self.Kd is not None:
            derivative = (self.error - self.prev_error)/self.dt
            D = self.Kd*derivative

        value = P+I+D
        self.prev_error = self.error

        return value

class PIDController:
    def __init__(self, setpoint, Kp, gues, *, Ki = None, Kd =None, dt = 1, min = None, max = None):
        self.pid = MyPID(setpoint, Kp=Kp, Ki=Ki, Kd=Kd, dt = dt)
        self.min = min
        self.max = max
        self.gues = gues

    def __call__(self, measured_value, gues, *args, **kwds):
        update = self.pid(measured_value)
        newValue = gues + update
        if (self.min is not None) and (self.max is not None):
            newValue = np.clip(newValue, a_min=self.min, a_max=self.max)
        return  [newValue, ]  

    def predict(self, measured_value, *args, **kwds) :
        nv = self.__call__(measured_value, self.gues, *args, **kwds)
        self.gues = nv[0]
        return nv