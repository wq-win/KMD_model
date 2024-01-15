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
        V, b = state
        # DV = Input - b
        # Db = 2 * V - 0.01 # -V0
        DV = (Input - b - 1*V)
        Db = 0.1*(Input - b) # -V0
        return [DV, Db]

    def step(self):
        state_new = rk4(self.dt, self.state, self.Input, self.derivative)
        self.last_state = copy.deepcopy(self.state)
        self.state = state_new
        return state_new

if __name__ == "__main__":
    kc_dan = KC_DAN()
    t = range(500)
    state_list = list()
    input_list = list()
    state_list_b = []
    for i in t:
        if i < 200:
            kc_dan.Input = 0.5 * i / 100
        elif 200 <= i < 300 :
            kc_dan.Input = 0
        elif 300 <= i < 325:
            kc_dan.Input = 4 * (i - 300) / 100
        # elif 325 <= i < 350:
        #     kc_dan.Input = 1 - 4 * (i -325) / 100
        elif 325<= i < 400:
            kc_dan.Input = 1    
        else:
            kc_dan.Input = 0
        input_list.append(kc_dan.Input)
        kc_dan.step()
        state_list.append(kc_dan.state[0])
        state_list_b.append(kc_dan.state[1])
    # plt.plot(t, np.exp(np.array(state_list)*10) /35)
    state_list_np = np.array(state_list)*10
    state_list_np[state_list_np<0]=0
    plt.figure()
    plt.plot(t, state_list_np)
    plt.plot(t, input_list)
    plt.plot(t,state_list_b)
    plt.show()
