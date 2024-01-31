from matplotlib import pyplot as plt
import numpy as np

def sigmoid(x):  
    return 1 / (1 + np.exp(-(np.array(x)-2)))

class KMDmodel():
    def __init__(self,n_K=6,n_M=3,n_D=3) -> None:
        self.task_input_KC = {'0':[0,1],'1':[2,3],'2':[4,5]}
        self.task_output_KC = {'0':[2,4],'1':[1,5],'2':[1,3]}
        self.potential_KC = {'0':[0,1],'1':[2,3],'2':[4,5]}
        self.KC = np.zeros(n_K,dtype=float)
        self.MBON = np.zeros(n_M,dtype=float)
        self.DAN = np.zeros(n_D,dtype=float)
        self.threshold = sigmoid(1)
        self.t = 0

    def activatedKC(self):
    # 返回到达阈值KC下标
        record = []
        self.KC = sigmoid(self.KC)
        for index,value in enumerate(self.KC):
            if value >= self.threshold:
                record.append(index)        
        return record
    
    def activatedMBON(self,inputKC,outputMBON):
        self.MBON = np.zeros(len(self.MBON))
        # self.DAN = np.zeros(len(self.DAN))
    # 将激活KC对应的MBON输出
        if set(inputKC).issubset(set(self.activatedKC())):
            self.MBON[outputMBON] = 1
            # self.DAN[outputMBON] = 1

    def activatedDAN(self,inputMBON,inputDAN):
        tokc = []
        pokc = []
        self.KC = np.zeros(len(self.KC))
        for i, value in enumerate(inputMBON):
            if value == 1:
                tokc = self.task_output_KC[str(i)]
        for i, value in enumerate(inputDAN):
            if value == 1:
                pokc = self.potential_KC[str(i)]        
        for i in tokc:
            self.KC[i] += 1
        for i in pokc:
            # self.KC[i] += 0.4
            self.KC[i] += 1
        self.KC = np.array(self.KC)
        self.KC[self.KC>1] = 1
         
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

    def step(self):
        n = self.t
        self.activatedMBON(self.task_input_KC[str(n)],n)
        self.activatedDAN(self.MBON,self.DAN)
        # print(self.KC,self.MBON,self.DAN)
        self.t += 1
        self.t %= 3   

kmd = KMDmodel()
kmd.KC = [1,1,0,0,0,0]
kmd.DAN = [0,1,0]
kc,mbon,dan = [],[],[]
for i in range(20):
    kc.append(kmd.KC)
    mbon.append(kmd.MBON)
    kmd.step()

plt.figure(1)
# plt.plot(range(100), mbon)
# plt.xlabel('all mbon')
# plt.show()
b=range(20)
mbon = np.array(mbon)
plt.subplot(311)
plt.plot(b, mbon.T[0])
plt.subplot(312)
plt.plot(b, mbon.T[1])
plt.subplot(313)
plt.plot(b, mbon.T[2])
plt.xlabel('mbon')
# plt.show()

kc = np.array(kc)
plt.figure(2)
for i in range(6):
    a = 611+i
    plt.subplot(a)
    plt.plot(b,kc.T[i])
plt.xlabel('kc')
plt.show()       
