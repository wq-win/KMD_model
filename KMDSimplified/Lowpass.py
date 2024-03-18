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

class LowPassFilter:
    """
    current输入当前膜电位,
    history是current历史轨迹,注意历史轨迹初值应该等于当前,
    derivative()是history动力学方程,
    step()返回history,
    update()更新current,
    """
    def __init__(self, current=None, history=None, tau=0.1, dt=0.1) -> None:
        if current is not None:
            self.current = current
        else:
            assert current is not None, "The current is None."
        if history is None:            
            self.history = current
        else:
            self.history = history
        self.tau = tau
        self.dt = dt
        self.currentnew = self.current
        self.historynew = self.history
    
    def derivative(self, state, inputs=0):
        history = state
        Dhistory = self.tau * (self.current - history)
        return Dhistory

    def step(self, dt, current, inputs=0 ):
        """
        返回新的历史轨迹
        """
        if dt is None:
            dt = self.dt
        state = self.history
        self.current = current
        statenew = rk4(dt, state, inputs, self.derivative)
        self.historynew = statenew
        self.currentnew = self.current
        return statenew

    def update(self):
        self.current = self.currentnew
        self.history = self.historynew
        
        
if __name__ == "__main__":
    lpf = LowPassFilter(current=0)
    hlist = []
    inputlist = []
    t = range(500)
    dt = 1
    for i in t:
        if i < 200:
            cur = np.array([0.5 * i / 100, 0.5 * i / 100 *2])
        elif 200 <= i < 300 :
            cur = np.array([0, 0])
        elif 300 <= i < 325:
            cur = np.array([4 * (i - 300) / 100, 4 * (i - 300) / 100 *2])
        elif 325<= i < 400:
            cur = np.array([1, 2])
        else:
            cur = np.array([0, 0])
        inputlist.append(cur)
        h = lpf.step(dt, cur)
        lpf.update()
        hlist.append(h)
    # print(np.array(hlist).shape, np.array(inputlist).shape)
    # print(hlist)
    plt.figure()
    plt.xlim(0,dt*500)
    plt.plot(t,inputlist)
    plt.plot(t,hlist)
    plt.show()