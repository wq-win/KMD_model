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
        # if phi_new > self.threshold:
        #     self.active_KC = True
        return phi_new


dt = 0.1
t_start = 0
t_end = 50
times = np.arange(t_start, t_end, dt)

phi = 0.1
threshold = 0.5
DAN = DAN_Neuron(phi=phi, threshold=threshold)
phi_state = []
for t in times:
    phi_state.append(phi)
    phi = DAN.step(phi, dt)
phi_state = np.array(phi_state)
phi_state[phi_state > threshold] = 1
plt.figure()
plt.plot(times, phi_state, )
plt.show()