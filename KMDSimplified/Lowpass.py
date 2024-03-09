from matplotlib import pyplot as plt
import numpy as np
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

class Lowpass:
    def __init__(self) -> None:
        self.history = 0
        self.current = 1
        self.tau = 0.1
    
    def derivative(self, state, inputs=0):
        history = state
        Dhistory = self.tau * (self.current - history)
        return Dhistory

    def step(self, dt, inputs=0):
        state = self.history
        statenew = rk4(dt, state, inputs, self.derivative)
        self.history = statenew
        return statenew

        
if __name__ == "__main__":
    lowpass = Lowpass()
    hlist = []
    inputlist = []
    t = range(500)
    for i in t:
        if i < 200:
            lowpass.current = 0.5 * i / 100
        elif 200 <= i < 300 :
            lowpass.current = 0
        elif 300 <= i < 325:
            lowpass.current = 4 * (i - 300) / 100
        elif 325<= i < 400:
            lowpass.current = 1
        else:
            lowpass.current = 0
        inputlist.append(lowpass.current)
        h = lowpass.step(1)
        hlist.append(h)

    # print(hlist)
    plt.figure()
    plt.plot(t,inputlist)
    plt.plot(t,hlist)
    plt.show()