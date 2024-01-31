import random
from matplotlib import pyplot as plt
import numpy as np

def sigmoid(x):  
    return 1 / (1 + np.exp(-(np.array(x)-2)))

class KMDmodel:
    def __init__(self,n_K=6,n_M=3,n_D=3) -> None:
        # 输入KC
        self.inputsensor = np.zeros(n_K,dtype=float)
        self.potentialKC = {'0':[0,1],'1':[2,3],'2':[4,5]}
        # 输入MBON
        self.KCtoMBON = {'0':[2,4],'1':[1,3],'2':[3,4]}
        # 初始化
        self.KC = np.zeros(n_K,dtype=float)
        self.MBON = np.zeros(n_M,dtype=float)
        self.DAN = np.zeros(n_D,dtype=float)

    def outputKC(self, inputsensor,inputDAN):
        pokc = []
        self.KC = np.array(inputsensor)
        for i, value in enumerate(inputDAN):
            if value == 1:
                pokc.append(self.potentialKC[str(i)])
        for i in pokc:
            for j in i:
                self.KC[j] += 1
        self.KC = sigmoid(self.KC)
        # return self.KC
    
    def outputMBON(self):
        a = []
        self.MBON = np.zeros(len(self.MBON))        
        for key in self.KCtoMBON:
            for i in self.KCtoMBON[key]:
                a.append(self.KC[i])
                if sum(a)>= 1:
                    self.MBON[int(key)] = 1
            a = []
        # return self.MBON
if __name__ == "__main__":
    kmd = KMDmodel()
    inputsensorlist = []
    KClist = []
    inputDANlist = []
    MBONlist = []
    plotlens = range(20)

    # 输入sensor
    for i in plotlens:
        temp = [random.choice([0,1]) for _ in range(6)]
        kmd.inputsensor = temp
        inputsensorlist.append(temp)
    print(inputsensorlist)
        
    # 输入DAN，输出KC，MBON
    for i in plotlens:
        kmd.DAN = [random.choice([0,1]) for _ in range(3)]
        # print(kmd.DAN)
        inputDANlist.append(kmd.DAN)
        # print(kmd.outputKC(kmd.inputsensor,kmd.DAN))
        kmd.outputKC(inputsensorlist[i],kmd.DAN)
        KClist.append(kmd.KC)
        kmd.outputMBON()
        MBONlist.append(kmd.MBON)

    # print(inputDANlist)
    # print(KClist)
    # print(MBONlist)

    inputsensorlist = np.array(inputsensorlist)
    KClist = np.array(KClist)
    inputDANlist = np.array(inputDANlist)
    MBONlist = np.array(MBONlist)

    plt.figure(1)
    for i in range(6):
        a = 611+i
        plt.subplot(a)
        plt.plot(plotlens,inputsensorlist.T[i])
    # plt.plot(plotlens,inputsensorlist)
    plt.xlabel('inputsensor')

    plt.figure(2)
    for i in range(6):
        a = 611+i
        plt.subplot(a)
        plt.plot(plotlens,KClist.T[i])
    # plt.plot(plotlens,KClist)
    plt.xlabel('KC')

    plt.figure(3)
    for i in range(3):
        a = 311+i
        plt.subplot(a)
        plt.plot(plotlens,inputDANlist.T[i])
    # plt.plot(plotlens,inputDANlist)
    plt.xlabel('DAN')

    plt.figure(4)
    for i in range(3):
        a = 311+i
        plt.subplot(a)
        plt.plot(plotlens,MBONlist.T[i])
    # plt.plot(plotlens,MBONlist)
    plt.xlabel('MBON')
    plt.show()
