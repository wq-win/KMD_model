import numpy as np
import matplotlib.pyplot as plt
import copy
def rk4(h, y, inputs, f):

    k1 = f(y, inputs)
    k2 = f([y[0] + h / 2 * k1[0], y[1] + h / 2 * k1[1]], inputs)
    k3 = f([y[0] + h / 2 * k2[0], y[1] + h / 2 * k2[1]], inputs)
    k4 = f([y[0] + h * k3[0], y[1] + h * k3[1]], inputs)

    y_new = [y[0] + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), 
             y[1] + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])]
    
    return y_new
class KC_DAN():
    def __init__(self, Input=1, Initstate=[0,0]):
        self.Input = Input
        self.state = Initstate
        self.last_state = Initstate
        self.dt = 1
    def derivative(self, state, Input):
        DV = Input - state[1]
        Db = state[0] - 0.1 #-V0
        return [DV, Db]

    def step(self):
        state_new = rk4(self.dt, self.state, self.Input, self.derivative)
        self.last_state = copy.deepcopy(self.state)
        self.state = state_new
        return state_new

if __name__ == "__main__":
    kc_dan = KC_DAN()
    t = range(100)
    state_list = list()
    input_list = list()
    for i in t:
        if i < 50:
            kc_dan.Input = 1/2
        else:
            kc_dan.Input = -i / 100 + 1
        input_list.append(kc_dan.Input)
        kc_dan.step()
        state_list.append(kc_dan.state[0])
    plt.plot(t, state_list)
    plt.plot(t, input_list)
    plt.show()
