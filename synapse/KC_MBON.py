from neuron_model.neuron_base import rk4
from synapse_base import Synapse


class KC_MBON(Synapse):
    def __init__(self,phi_K=0.1,phi_M=0.1,D=0.1):
        super().__init__(initial_weight=0.5)
        self.W_MK = self.weight
        self.phi_K = phi_K
        self.phi_M = phi_M
        self.D = D
        self.episilon = 0.01

    def derivative(self, state, inputs=0):
        W_MK = state  # 怎么没用到这个变量
        DW_MK = self.episilon * self.phi_K * self.phi_M * self.D
        return DW_MK

    def step(self, state, dt, inputs=0):
        W_MK_new = rk4(dt, state, inputs,self.derivative)
        return W_MK_new


