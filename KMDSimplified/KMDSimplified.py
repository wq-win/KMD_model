import numpy as np
import matplotlib.pyplot as plt
def sigmoid(input):
    return 1 / (1+np.exp(-input))

class KMDSimple():
    def __init__(self, NumberofTask=3, t=0, dt=1):
        # self.TaskList = ["t1", "t3", "t2"]
        # self.tlist = [3,2,4]
        self.KC = [1,1,0,0,0,0]
        self.KC_add = [0] * 6
        self.MBON = [0,0,0]
        self.DAN = [0,0,0]
        self.threshold = 1
        # self.dt = dt
        self.t = t

    def isthreshold(self):
        record = list()
        for i, KC in enumerate(self.KC):
            if KC >= self.threshold:
                record.append(i+1)
            else: continue
        return record

    def KC_MBON_DAN(self):
        if 1 in self.isthreshold() and 2 in self.isthreshold():
            self.MBON[0] = 1
        if 2 in self.isthreshold() and 6 in self.isthreshold():
            self.MBON[1] = 1
        if 3 in self.isthreshold() and 5 in self.isthreshold():
            self.MBON[2] = 1
        return self.MBON
        
    def step(self):
        self.t += 1
        self.KC_MBON_DAN()
        if self.MBON == [1,0,0] and self.t == 3:
            self.t = 0
            self.MBON = [0,0,0]
            self.DAN = [1,0,0]
            self.KC_add = [0,0.4,1,0.4,1,0]
            self.KC = [min(1,self.KC[i] + self.KC_add[i]) for i in range(6)]
            self.KC[0] = 0
        else:
            for i, item in enumerate(self.KC):
                if 0 < item<1:
                    item -= 0.1

        if self.MBON == [0,1,0] and self.t == 2:
            self.t = 0
            self.MBON = [0,0,0]
            self.DAN = [0,1,0]
            self.KC_add = [1,1,0,0.4,0.4,0]
            self.KC = [min(1, self.KC[i] + self.KC_add[i]) for i in range(6)]
            self.KC[5] = 0
        else:
            for i, item in enumerate(self.KC):
                if 0 < item<1:
                    item -= 0.1
                    
        if self.MBON == [0,0,1] and self.t == 4:
            self.t = 0
            self.MBON = [0,0,0]
            self.DAN = [0,0,1]
            self.KC_add = [0,1,0,0.4,0.4,1]
            self.KC = [min(1, self.KC[i] + self.KC_add[i]) for i in range(6)]
            self.KC[2] = 0
        else:
            for i, item in enumerate(self.KC):
                if 0 < item<1:
                    item -= 0.1


if __name__ == "__main__":

    kmd = KMDSimple()
    mbon = list()
    dan = list()
    kc = list()
    for i in range(100):
        mbon.append(kmd.MBON)
        kc.append(kmd.KC)
        kmd.step()
    plt.figure(1)
    plt.plot(range(100), mbon)
    plt.show()

    plt.figure(2)
    plt.plot(range(100),kc)
    plt.show()
        
            

        