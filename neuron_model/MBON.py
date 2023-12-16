import numpy as np
import matplotlib.pyplot as plt

from neuron_model.neuron_base import Neuron, rk4


class MBON_Neuron(Neuron):
    def __init__(self, phi=0.1, threshold=0.5):
        # todo 修改初始化赋值
        super().__init__(phi, threshold)
        self.MBON = 1  # 当前兴奋程度

    def derivative(self, state, inputs=0):
        phi = state
        Dphi = self.alpha * self.MBON - self.beta * phi
        return Dphi

    def step(self, state, dt, inputs=0):
        phi_new = rk4(dt, state, inputs, self.derivative)
        # todo 阈值重新考虑
        # if phi_new > self.threshold:
        return phi_new


dt = 0.1
t_start = 0
t_end = 50
times = np.arange(t_start, t_end, dt)

phi = 0.1
threshold = 0.5
MBON = MBON_Neuron(phi=phi, threshold=threshold)
phi_state = []
for t in times:
    phi_state.append(phi)
    phi = MBON.step(phi,dt)
phi_state = np.array(phi_state)
phi_state[phi_state > threshold] = 1
plt.figure()
plt.plot(times,phi_state,)
plt.show()