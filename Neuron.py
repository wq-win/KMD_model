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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self,phi=0.1, threshold=0.5):
        self.phi = phi  # 历史兴奋程度,气体数量
        self.alpha = 0.1  # 进气
        self.N = 1  # 当前兴奋程度
        self.beta = 0.1  # 漏气
        self.threshold = threshold  # 激活输出的阈值

    def derivative(self, state, inputs=0):
        phi = state
        Dphi = self.alpha * self.N - self.beta * phi
        return Dphi

    def step(self, state, dt, inputs=0):
        phi_new = rk4(dt, state, inputs, self.derivative)
        return phi_new

