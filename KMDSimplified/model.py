from matplotlib import pyplot as plt
import numpy as np


class KMDmodel():
    def __init__(self,n_K=6,n_M=3,n_D=3) -> None:
        self.task_input_KC = {'0':[0,1],'2':[1,5],'1':[2,4]}
        self.task_output_KC = {'0':[1,2,4],'2':[0,4,1],'1':[1,4,5]}
        self.potential_KC = {'0':[3,4],'1':[1,3],'2':[1,3]}
        self.KC = np.zeros(n_K,dtype=float)
        self.MBON = np.zeros(n_M,dtype=float)
        self.DAN = np.zeros(n_D,dtype=float)
        self.threshold = 1
        self.t = 0

    def activatedKC(self):
    # 返回到达阈值KC下标
        record = []
        for index,value in enumerate(self.KC):
            if value >= self.threshold:
                record.append(index)        
        return record
    
    def activatedMBON(self,inputKC,outputMBON):
        self.MBON = np.zeros(len(self.MBON))
        self.DAN = np.zeros(len(self.DAN))
    # 将激活KC对应的MBON输出
        if set(inputKC).issubset(set(self.activatedKC())):
            self.MBON[outputMBON] = 1
            self.DAN[outputMBON] = 1

    def activatedDAN(self,inputMBON):
        tokc = []
        pokc = []
        self.KC = np.zeros(len(self.KC))
        # self.MBON = np.zeros(len(self.MBON))
        for i, value in enumerate(inputMBON):
            if value == 1:
                tokc = self.task_output_KC[str(i)]
                pokc = self.potential_KC[str(i)]
        for i in tokc:
            self.KC[i] = 1
        for i in pokc:
            self.KC[i] += 0.4
        self.KC = np.array(self.KC)
        self.KC[self.KC>1] = 1
         

    def step(self):
        n = self.t
        self.activatedMBON(self.task_input_KC[str(n)],n)
        self.activatedDAN(self.MBON)
        # print(self.KC,self.MBON,self.DAN)
        self.t += 1
        self.t %= 3   
        # print(n)     

kmd = KMDmodel()
kmd.KC = [1,1,0,0,1,1]

kc,mbon,dan = [],[],[]
for i in range(100):
    kc.append(kmd.KC)
    mbon.append(kmd.MBON)
    kmd.step()

# plt.figure(1)
# plt.plot(range(100), mbon)

# mbon = np.array(mbon)
# plt.subplot(311)
# plt.plot(range(100), mbon.T[0])
# plt.subplot(312)
# plt.plot(range(100), mbon.T[1])
# plt.subplot(313)
# plt.plot(range(100), mbon.T[2])

kc = np.array(kc)
plt.figure(2)
for i in range(6):
    a = 611+i
    plt.subplot(a)
    plt.plot(range(100),kc.T[i])
plt.show()       
