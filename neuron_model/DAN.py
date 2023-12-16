import numpy as np
from matplotlib import pyplot as plt

from neuron_model.neuron_base import Neuron, rk4


class DAN_Neuron(Neuron):
    def __init__(self, phi=0.1, threshold=0.5):
        # todo 修改初始化赋值
        super().__init__(phi, threshold)
        self.DAN = 1  # 当前兴奋程度
        self.active_KC = False

    def derivative(self, state, inputs=0):
        phi = state
        Dphi = self.alpha * self.DAN - self.beta * phi
        return Dphi

    def step(self, state, dt, inputs=0):
        phi_new = rk4(dt, state, inputs, self.derivative)
        # todo 阈值重新考虑
        return phi_new

