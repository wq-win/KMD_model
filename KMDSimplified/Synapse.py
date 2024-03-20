from Neuron import Neuron, rk4
import numpy as np
from matplotlib import pyplot as plt


class Synapse:
    def __init__(self, preNeuron: Neuron, postNeuron: Neuron, weight=None, reward=0, alpha=1, beta=1, dt=0.01, tau=1,) -> None:
        self.preNeuron = preNeuron
        self.postNeuron = postNeuron
        self.preNum = self.preNeuron.num
        self.postNum = self.postNeuron.num
        self.preI = self.preNeuron.I
        self.postI = self.postNeuron.I
        self.prePotential = self.preNeuron.potential
        self.prePotentialnew = self.prePotential
        self.preTrajectory = self.prePotential  # self.preNeuron.trajectory
        self.preTrajectorynew = self.preTrajectory
        self.postPotential = self.postNeuron.potential
        self.postPotentialnew = self.postPotential
        self.postTrajectory = self.postPotential  # self.preNeuron.trajectory
        self.postTrajectorynew = self.postTrajectory
        if weight is not None:
            self.weight = weight
        else:
            assert weight is not None, "The weight is None."
        assert self.weight.shape == (
            self.postNum, self.preNum), "The shape of the weight does not match. "
        self.reward = np.array(reward)
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.tau = tau

    def derivative(self, state, inputs=0):
        w = state
        Dw = (np.matmul(self.beta * np.tile(self.postTrajectory, self.postNum),
              (self.alpha * self.preTrajectory.T - w)) * self.reward) / self.tau
        return Dw

    def step(self, dt, preI, inputs=0):
        self.preI = preI
        if dt is None:
            dt = self.dt
        state = self.weight
        print('synapse',dt)
        statenew = rk4(dt, state, inputs, self.derivative)
        self.weight = statenew
        self.preTrajectorynew = self.preNeuron.step(dt, self.preI)
        self.postI = np.matmul(self.weight, self.prePotential)
        self.postTrajectorynew = self.postNeuron.step(dt, self.postI)
        return statenew

    def update(self):
        self.prePotential = self.preNeuron.potential
        self.preTrajectory = self.preTrajectorynew
        self.postPotential = self.postNeuron.potential
        self.postTrajectory = self.postTrajectorynew
        # self.preNeuron.potential = self.prePotentialnew
        # self.postNeuron.potential = self.postPotential
        self.preNeuron.update()
        self.postNeuron.update()


class SynapseKM(Synapse):
    def __init__(self, preNeuron: Neuron, postNeuron: Neuron, weight=None, reward=0, alpha=1, beta=1, dt=0.01, tau=1) -> None:
        super().__init__(preNeuron, postNeuron, weight, reward, alpha, beta, dt, tau)

    def derivative(self, state, inputs=0):
        return super().derivative(state, inputs)

    def step(self, dt, preI, inputs=0):
        return super().step(dt, preI, inputs)

    def update(self):
        return super().update()


class SynapseKD(Synapse):
    def __init__(self, preNeuron: Neuron, postNeuron: Neuron, weight=None, reward=0, alpha=1, beta=1, dt=0.01, tau=1) -> None:
        super().__init__(preNeuron, postNeuron, weight, reward, alpha, beta, dt, tau)
        self.delta = self.preNeuron.delta

    def derivative(self, state, inputs=0):
        w = state
        delta = self.delta
        Dw = (np.matmul(self.beta * np.tile(self.postPotential, self.postNum),
              (self.alpha * delta.T - w)) * self.reward) / self.tau
        return Dw

    def step(self, dt, preI, inputs=0):
        return super().step(dt, preI, inputs)

    def update(self):
        self.preNeuron.trajectory = self.preTrajectorynew
        self.delta = self.preNeuron.updateDelta()
        return super().update()


class SynapseDK(Synapse):
    def __init__(self, preNeuron: Neuron, postNeuron: Neuron, weight=None, reward=0, alpha=1, beta=1, dt=0.01, tau=1) -> None:
        super().__init__(preNeuron, postNeuron, weight, reward, alpha, beta, dt, tau)
        self.delta = self.postNeuron.delta

    def derivative(self, state, inputs=0):
        w = state
        delta = self.delta
        Dw = (np.matmul(self.beta * np.tile(delta, self.postNum),
              (self.alpha * self.preTrajectory.T - w)) * self.reward) / self.tau
        return Dw

    def step(self, dt, preI, inputs=0):
        return super().step(dt, preI, inputs)

    def update(self):
        self.delta = self.postNeuron.updateDelta()
        return super().update()


# t = range(100)
sim = 60
start = 0
dt = 1
end = start + sim * dt
t = np.linspace(start, end, sim)

numKC, numMBON, numDAN = 6, 3, 3
# 测试synapseKM
initWeightKM = np.ones([numMBON, numKC]) * .1
IKC = np.ones(numKC).reshape(numKC, 1)

# IMBON = np.ones(numMBON).reshape(numMBON,1)
KC = Neuron(numKC, IKC,dt=dt)
IMBON = np.matmul(initWeightKM, KC.potential)
MBON = Neuron(numMBON, IMBON,dt=dt)

skm = SynapseKM(KC, MBON, initWeightKM, reward=1, dt=dt)
skmweight, KCp, KCt, MBONp, MBONt = [], [], [], [], []
KCI, MBONI = [], []
for i in t:
    KCI.append(skm.preI[0][0])
    MBONI.append(skm.postI[0][0])
    KCp.append(skm.prePotential[0][0])
    KCt.append(skm.preTrajectory[0][0])
    MBONp.append(skm.postPotential[0][0])
    MBONt.append(skm.postTrajectory[0][0])
    KMWeight = skm.step(dt, np.ones(numKC).reshape(numKC, 1))
    skm.update()
    skmweight.append(KMWeight[0][0])


# print(MBONp[:10])
# print(MBONp[::-1][:10])
plt.xlim(start, dt*sim)
plt.plot(t, KCI, label='KCI')
plt.plot(t, MBONI, label='MBONI')
plt.plot(t, skmweight, label='skmweight')
plt.plot(t, KCp, label='KCp')
plt.plot(t, KCt, label='KCt', linestyle='--')
plt.plot(t, MBONp, label='MBONp')
plt.plot(t, MBONt, label='MBONt')
plt.legend()
plt.show()
