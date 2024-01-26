import random
from matplotlib import pyplot as plt
import numpy as np
def rk4(h, y, inputs, f):

    k1 = f(y, inputs)
    k2 = f([y[0] + h / 2 * k1[0], y[1] + h / 2 * k1[1]], inputs)
    k3 = f([y[0] + h / 2 * k2[0], y[1] + h / 2 * k2[1]], inputs)
    k4 = f([y[0] + h * k3[0], y[1] + h * k3[1]], inputs)

    y_new = [y[0] + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), 
             y[1] + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])]
    
    return y_new
def rk4(h, y, inputs, f):
    '''
    用于数值积分的rk4函数。
    args:
        h - 步长
        y - 当前状态量
        inputs - 外界对系统的输入
        f - 常微分或偏微分方程
    return:
        y_new - 新的状态量,即经过h时间之后的状态量
    '''
    k1 = f(y, inputs)
    k2 = f(y + h / 2 * k1, inputs)
    k3 = f(y + h / 2 * k2, inputs)
    k4 = f(y + h * k3, inputs)

    y_new = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_new


class Synapse:
    def __init__(self,n_preneuron,n_postneuron) -> None:
        self.n_preneuron = n_preneuron
        self.n_postneuron = n_postneuron
        self.preneuron = np.array([random.choice([0,1]) for _ in range(self.n_preneuron)])
        self.postneuron = np.array([random.choice([0,1]) for _ in range(self.n_postneuron)])
        self.weight = np.zeros([self.n_preneuron, self.n_postneuron])
        self.isreward = np.array(random.choice([0,1]))
        # 超参数
        self.alpha = 0.5
        self.tau = 0.5
        self.dt = 1

    def derivative(self, state, inputs=0):
        w = state
        Dw = np.zeros_like(self.weight)
        for r in range(self.weight.shape[0]):
            for c in range(self.weight.shape[1]):
                Dw[r][c] = (self.preneuron[r] * self.postneuron[c] - w[r][c]) * self.isreward
        return Dw        
            
        
    def step(self, state, dt, inputs=0):
        statenew = rk4(dt, state, inputs, self.derivative)
        return statenew
        
KCtoMBON = Synapse(6,3)
w = KCtoMBON.weight
KCtoMBON.step(w,1)

