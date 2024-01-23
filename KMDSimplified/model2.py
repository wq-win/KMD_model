from matplotlib import pyplot as plt
import numpy as np

def sigmoid(x):  
    return 1 / (1 + np.exp(-(np.array(x)-2)))

class KMDmodel():
    def __init__(self,n_K=6,n_M=3,n_D=3) -> None:
        # 输入KC
        self.inputsensor = np.zeros(n_K,dtype=float)
        self.potentialKC = {'0':[0,1],'1':[2,3],'2':[4,5]}
        # 输入MBON
        self.inputKC = {'0':[2,4],'1':[1,3],'2':[3,4]}
        # 初始化
        self.KC = np.zeros(n_K,dtype=float)
        self.MBON = np.zeros(n_M,dtype=float)
        self.DAN = np.zeros(n_D,dtype=float)

    def outputKC(self, inputsensor,inputDAN):
        pokc = []
        self.KC = inputsensor
        for i, value in enumerate(inputDAN):
            if value == 1:
                pokc.append(self.potentialKC[str(i)])
        for i in pokc:
            for j in i:
                self.KC[j] += 1
        self.KC = sigmoid(self.KC)
        return self.KC
    
    def outputMBON(self):
        a = []
        for key in self.inputKC:
            for i in self.inputKC[key]:
                a.append(self.KC[i])
                if sum(a)>= 1:
                    self.MBON[int(key)] = 1
            a = []
        return self.MBON
kmd = KMDmodel()
kmd.inputsensor = [0,1,0,1,1,0]
kmd.DAN = [1,1,1]
print(kmd.outputKC(kmd.inputsensor,kmd.DAN))
print(kmd.outputMBON())
