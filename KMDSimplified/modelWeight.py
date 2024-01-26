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

        
    def step(self):
        s = self.weight.shape
        for r in range(s[0]):
            for c in range(s[1]):
                self.weight[r][c] = (self.preneuron[r] * self.postneuron[c] - self.weight[r][c]) * self.isreward

        
KCtoMBON = Synapse(6,3)
simlen = range(100)
weightlist = []
for i in simlen:
    KCtoMBON.preneuron = np.array([random.choice([0,1]) for _ in range(6)])
    KCtoMBON.postneuron = np.array([random.choice([0,1]) for _ in range(3)])
    KCtoMBON.isreward = np.array(random.choice([0,1]))
    weightlist.append(KCtoMBON.weight[0][0])
    KCtoMBON.step()
# print(KCtoMBON.weight)
plt.plot(simlen,weightlist)
plt.show()