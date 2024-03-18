import numpy as np
from matplotlib import pyplot as plt
from Lowpass import LowPassFilter


def sigmoid(x):
    return 2 * (1 / (1 + np.exp(-(np.array(x)))) - 0.5)


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


class Neuron:
    def __init__(self, num=3, I=0, potential=None, trajectory=None, tau=0.1, dt=1) -> None:
        """
        根据输入I,获取神经元激活的兴奋程度potential(兴奋当前1),
        trajectory(兴奋历史2)是potential的历史兴奋程度,
        delta(兴奋增量3)是兴奋当前-兴奋历史(1-2),
        updateDelta()更新delta,
        ,
        """
        self.num = num
        self.I = I
        assert self.num == np.size(I), "The input size is different from the neuron number."
        if potential is None:
            self.potential = self.activation(I)
        else:
            self.potential = potential
        if trajectory is None:
            self.trajectory = self.potential
        self.potentialnew = self.potential
        self.trajectorynew = self.trajectory
        self.delta = self.potential - self.trajectory        
        self.tau = tau
        self.dt = dt
        self.lowpassfilter = LowPassFilter(self.potential, self.trajectory, self.tau, self.dt)
    
    def updateDelta(self):
        self.delta = self.potential - self.trajectory 
        return self.delta
    
    def activation(self, I):
        return sigmoid(I)    
    """
    # def step(self, dt, inputs=0):
    #     if dt is None:
    #         dt = self.dt
    #     state = self.history
    #     statenew = rk4(dt, state, inputs, self.derivative)
    #     self.history = statenew
    #     return statenew
    """
    def step(self, dt, I):
        if dt is None:
            dt = self.dt
        assert self.num == np.size(I), "The input size is different from the neuron number."
        self.potential = self.activation(I)
        self.trajectory = self.potential
        self.trajectorynew = self.lowpassfilter.step(dt=dt, current=self.potential)
        self.potentialnew = self.potential
        return self.trajectorynew

    def update(self):
        self.potential = self.potentialnew
        self.trajectory = self.trajectorynew
        self.updateDelta()
        self.lowpassfilter.update()
        

if __name__ == "__main__":
    t = range(500)
    # 测试Neuron
    num = 6
    I = np.zeros(num).reshape(num,1)
    KC = Neuron(num,I,tau=0.1)
    plist, tlist = [], []
    for i in t:
        if i < 200:
            I = np.array([[0.5 * i / 100, 0.5 * i / 100 *2, 1.5 * i / 100, 0.5 * i / 100 *4, 2.5 * i / 100, 0.5 * i / 100 *6]])
        elif 200 <= i < 300 :
            I = np.array([[0, 0, 0, 0, 0, 0]])
        elif 300 <= i < 325:
            I = np.array([[4 * (i - 300) / 100, 4 * (i - 300) / 100 *2, 4 * (i - 300) / 100, 4 * (i - 300) / 100 *2, 4 * (i - 300) / 100, 4 * (i - 300) / 100 *2]])
        elif 325<= i < 400:
            I = np.array([[1, 2, 3, 4, 5, 6]])
        else:
            I = np.array([[0, 0, 0, 0, 0, 0]])

        tra = KC.step(1, I)
        # print(tra)
        plist.append(KC.potential[0])
        KC.update()
        tlist.append(tra[0])  # 不知道为什么正常应该是[],结果实际返回[[]]   
    plt.figure()
    plt.plot(t, plist)
    plt.plot(t, tlist, '--')
    # plt.plot(t,np.array(plist).T[0])
    # plt.plot(t,np.array(tlist).T[0][0])
    # plt.plot(t,np.array(plist).T[1])
    # plt.plot(t,np.array(tlist).T[1][0])
    plt.show()    

    
