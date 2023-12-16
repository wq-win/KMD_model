import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib as mpl
from collections import deque
import time
import dill
from cycler import cycler
import os
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import butter, lfilter, freqz


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


class Synapse:
    def __init__(self, initial_weight=0.5):
        self.weight = initial_weight
        self.psp = 0.0  # postsynaptic potential，后突触电位

    def transmit(self, pre_synaptic_spike):
        """
        通过突触传递信号，计算后突触电位。

        参数：
        - pre_synaptic_spike: 前突触神经元的脉冲信号（1表示有脉冲，0表示没有）

        返回：
        - postsynaptic_potential: 后突触电位
        """
        self.psp = self.weight * pre_synaptic_spike
        return self.psp

    def update_weight(self, delta_weight):
        """
        更新突触权重。

        参数：
        - delta_weight: 权重的变化量
        """
        self.weight += delta_weight


class DynamicSynapseArray:
    def __init__(self, NumberOfSynapses=[1, 3], Period=None, tInPeriod=None, PeriodVar=None, \
                 Amp=None, WeightersCentre=None, WeightersCentreUpdateRate=0.000012, WeightersOscilateDecay=0.0000003, \
                 ModulatorAmount=0, InitAmp=0.4, t=0, dt=1, NormalizedWeight=False):

        self.NumberOfSynapses = NumberOfSynapses  # [1]
        self.dt = dt
        self.t = t
        self.PeriodCentre = np.ones(NumberOfSynapses).astype(
            np.float) * Period if Period is not None else 1000 + 100 * (np.random.rand(*NumberOfSynapses) - 0.5)
        self.Period = copy.deepcopy(self.PeriodCentre)
        self.tInPeriod = np.ones(NumberOfSynapses).astype(
            np.float) * tInPeriod if tInPeriod is not None else np.random.rand(*NumberOfSynapses) * self.Period
        self.PeriodVar = np.ones(NumberOfSynapses).astype(np.float) * PeriodVar if PeriodVar is not None else np.ones(
            NumberOfSynapses).astype(np.float) * 0.1
        self.Amp = np.ones(NumberOfSynapses).astype(np.float) * Amp if Amp is not None else np.ones(
            NumberOfSynapses).astype(np.float) * 0.2
        self.WeightersCentre = np.ones(NumberOfSynapses) * WeightersCentre if WeightersCentre is not None else (
                                                                                                                       np.random.rand(
                                                                                                                           *NumberOfSynapses) - 0.5) * InitAmp
        self.NormalizedWeight = NormalizedWeight
        print(self.WeightersCentre)
        print(np.sum(self.WeightersCentre, axis=1))
        if NormalizedWeight:
            self.WeightersCentre /= np.sum(self.WeightersCentre, axis=1)[:, None]
        self.WeightersCentreUpdateRate = np.ones(
            NumberOfSynapses) * WeightersCentreUpdateRate if WeightersCentreUpdateRate is not None else np.ones(
            NumberOfSynapses) * 0.000012
        self.Weighters = self.WeightersCentre + self.Amp * np.sin(self.tInPeriod / self.Period * 2 * np.pi)
        self.WeightersLast = copy.deepcopy(self.Weighters)
        self.WeightersOscilateDecay = np.ones(
            NumberOfSynapses) * WeightersOscilateDecay if WeightersOscilateDecay is not None else np.ones(
            NumberOfSynapses)
        self.ModulatorAmount = np.ones(NumberOfSynapses) * ModulatorAmount
        self.ZeroCross = np.ones(NumberOfSynapses, dtype=bool)

    def StepSynapseDynamics(self, dt, t, ModulatorAmount, PreSynActivity=None):

        if dt is None:
            dt = self.dt
        self.t = t
        self.tInPeriod += dt
        self.Weighters = self.WeightersCentre + self.Amp * np.sin(self.tInPeriod / self.Period * 2 * np.pi)
        self.WeightersCentre += (
                                        self.Weighters - self.WeightersCentre) * ModulatorAmount * self.WeightersCentreUpdateRate * dt
        if self.NormalizedWeight:
            self.WeightersCentre /= np.sum(np.abs(self.WeightersCentre), axis=1)[:, None]
        self.ModulatorAmount = np.ones(self.NumberOfSynapses) * ModulatorAmount
        self.Amp *= np.exp(-self.WeightersOscilateDecay * self.ModulatorAmount * dt)
        self.ZeroCross = np.logical_and(np.less(self.WeightersLast, self.WeightersCentre),
                                        np.greater_equal(self.Weighters, self.WeightersCentre))
        self.tInPeriod[self.ZeroCross] = self.tInPeriod[self.ZeroCross] % self.Period[self.ZeroCross]
        #        self.Period[self.ZeroCross] += (np.random.rand(*self.NumberOfSynapses)[self.ZeroCross]-0.5)*self.PeriodVar[self.ZeroCross]*self.Period[self.ZeroCross]+(self.PeriodCentre-self.Period)[self.ZeroCross]*0.03
        self.Period[self.ZeroCross] = np.random.normal(loc=self.PeriodCentre[self.ZeroCross],
                                                       scale=self.PeriodCentre[self.ZeroCross] * 0.1)
        self.WeightersLast = self.Weighters
        return self.Weighters

    def InitRecording(self):
        self.RecordingState = True
        self.Trace = {'Weighters': deque(),
                      'WeightersCentre': deque(),
                      'ModulatorAmount': deque(),
                      'Amp': deque(),
                      'Period': deque(),
                      'tInPeriod': deque(),
                      't': deque()
                      }

    def Recording(self):
        Temp = None
        for key in self.Trace:
            exec("Temp = self.%s" % (key))
            self.Trace[key].append(copy.deepcopy(Temp))

    def plot(self, path='', savePlots=False, StartTimeRate=0.3, DownSampleRate=10, linewidth=1, FullScale=False,
             NameStr=None, NeuronID=0):
        mpl.rcParams['axes.prop_cycle'] = cycler('color',
                                                 ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                                                  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'b', 'k'])
        if NameStr is None:
            NameStr = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        TracetInS = np.array(self.Trace['t'])[::DownSampleRate].astype(float) / 1000
        NumberOfSteps = len(TracetInS)
        if StartTimeRate == 0:
            StartStep = 0
        else:
            StartStep = NumberOfSteps - int(NumberOfSteps * StartTimeRate)
        FigureDict = {}
        FigureDict['ASynapse'] = plt.figure()
        SynapseID = 0
        figure0lines0, = plt.plot(TracetInS,
                                  np.array(self.Trace['ModulatorAmount'])[::DownSampleRate, NeuronID, SynapseID],
                                  linewidth=1)
        figure0lines1, = plt.plot(TracetInS,
                                  np.array(self.Trace['WeightersCentre'])[::DownSampleRate, NeuronID, SynapseID],
                                  linewidth=1)
        figure0lines2, = plt.plot(TracetInS, np.array(self.Trace['Weighters'])[::DownSampleRate, NeuronID, SynapseID],
                                  linewidth=1)
        plt.legend([figure0lines2, figure0lines1, figure0lines0],
                   ['Weight Fluctuation', 'Fluctuation Centre', 'Modulator Amount', ], loc=4)
        plt.title('Example Dynamics of a Synapse')
        FigureDict['Weighters'] = plt.figure()
        labels = [str(i) for i in range(self.Trace['Weighters'][0].shape[1])]
        figure1lines = plt.plot(TracetInS, np.array(self.Trace['Weighters'])[::DownSampleRate, NeuronID], label=labels,
                                linewidth=linewidth)
        plt.legend(figure1lines, labels)
        plt.xlabel('Time (s)')
        plt.title('Instantaneous Synaptic Strength')
        X = np.array(self.Trace['Weighters'])[StartStep:NumberOfSteps][::DownSampleRate, NeuronID, 0]
        Y = np.array(self.Trace['Weighters'])[StartStep:NumberOfSteps][::DownSampleRate, NeuronID, 1]
        Z = np.array(self.Trace['Weighters'])[StartStep:NumberOfSteps][::DownSampleRate, NeuronID, 2]
        FigureDict['2Weighters'] = plt.figure()
        plt.plot(X, Y)
        plt.xlabel('Time (s)')
        plt.title('2 Instantaneous Synaptic Strength')
        plt.xlabel('Instantaneous Synaptic Strength 0')
        plt.ylabel('Instantaneous Synaptic Strength 1')
        FigureDict['3Weighters'] = plt.figure()
        ax = FigureDict['3Weighters'].add_subplot(111, projection='3d')
        ax.plot(X, Y, zs=Z)
        ax.set_xlabel('Instantaneous Synaptic Strength 0')
        ax.set_ylabel('Instantaneous Synaptic Strength 1')
        ax.set_zlabel('Instantaneous Synaptic Strength 2')
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w', linewidth=linewidth)
        if FullScale:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
        FigureDict['WeightersCentre'] = plt.figure()
        figure4lines = plt.plot(TracetInS, np.array(self.Trace['WeightersCentre'])[::DownSampleRate, NeuronID],
                                label=labels, linewidth=linewidth)
        plt.legend(figure4lines, labels)
        plt.title('Center of Synaptic Strength Oscillation')
        plt.xlabel('Time (s)')
        FigureDict['Period'] = plt.figure()
        figure5lines = plt.plot(TracetInS, np.array(self.Trace['Period'])[::DownSampleRate, NeuronID], label=labels,
                                linewidth=linewidth)
        plt.legend(figure5lines, labels)
        plt.title('Period')
        plt.xlabel('Time (s)')
        plt.xlabel('Period (s)')
        FigureDict['tInPeriod'] = plt.figure()
        figure6lines = plt.plot(TracetInS, np.array(self.Trace['tInPeriod'])[::DownSampleRate, NeuronID], label=labels,
                                linewidth=linewidth)
        plt.legend(figure6lines, labels)
        plt.title('tInPeriod')
        plt.xlabel('Time (s)')
        plt.xlabel('Period (s)')
        FigureDict['Amp'] = plt.figure()
        figure6lines = plt.plot(TracetInS, np.array(self.Trace['Amp'])[::DownSampleRate, NeuronID], label=labels,
                                linewidth=linewidth)
        plt.legend(figure6lines, labels)
        plt.title('Amp')
        plt.xlabel('Time (s)')
        FigureDict['ModulatorAmount'] = plt.figure()
        figure7lines = plt.plot(TracetInS, np.array(self.Trace['ModulatorAmount'])[::DownSampleRate, NeuronID],
                                label=labels, linewidth=linewidth)
        plt.legend(figure7lines, labels)
        plt.xlabel('Time (s)')
        plt.title('ModulatorAmount')
        FigureDict['PoincareMap'] = plt.figure()
        figure8ax1 = FigureDict['PoincareMap'].add_subplot(111)
        points0, points1 = CrossAnalysis(np.array(self.Trace['Weighters'])[::DownSampleRate, NeuronID, 0],
                                         np.array(self.Trace['WeightersCentre'])[::DownSampleRate, NeuronID, 0],
                                         np.array(self.Trace['Weighters'])[::DownSampleRate, NeuronID], TracetInS)
        if FullScale:
            figure8ax1.set_xlim(0, 1)
            figure8ax1.set_ylim(0, 1)
        print('points0')
        print(points0['points'])
        print('points1')
        print(points1['points'])
        pointsploted0 = figure8ax1.scatter(points0['points'][:, 1], points0['points'][:, 2], c=points0['t'],
                                           cmap=plt.cm.get_cmap('Greens'), marker=".",
                                           edgecolor='none')  # c=c, ,  cmap=cm
        pointsploted1 = figure8ax1.scatter(points1['points'][:, 1], points1['points'][:, 2], c=points1['t'],
                                           cmap=plt.cm.get_cmap('Blues'), marker=".", edgecolor='none')
        plt.colorbar(pointsploted0)
        plt.colorbar(pointsploted1)
        plt.title('Poincare map')
        plt.xlabel('Instantaneous Synaptic Strength 1')
        plt.ylabel('Instantaneous Synaptic Strength 2')
        if savePlots == True:
            if not os.path.exists(path):
                os.makedirs(path)
            pp = PdfPages(path + "DynamicSynapse" + NameStr + '.pdf')
            for key in FigureDict:
                FigureDict[key].savefig(pp, format='pdf')
            pp.close()
        return FigureDict, ax


def CrossAnalysis(Oscillate, Reference, OscillateArray, Tracet):
    points0 = {'t': [], 'points': []}
    points1 = {'t': [], 'points': []}
    GreaterThanCentre = (Oscillate[0] > Reference[0])
    print(Oscillate[0])
    print(Reference[0])
    for i1 in range(len(Oscillate)):
        if GreaterThanCentre == True:
            if Oscillate[i1] < Reference[i1]:
                points0['points'].append(OscillateArray[i1])
                points0['t'].append(Tracet[i1])
                GreaterThanCentre = False
        elif GreaterThanCentre == False:
            if Oscillate[i1] > Reference[i1]:
                points1['points'].append(OscillateArray[i1])
                points1['t'].append(Tracet[i1])
                GreaterThanCentre = True
    points0['points'] = np.array(points0['points'])
    points1['points'] = np.array(points1['points'])
    points0['t'] = np.array(points0['t'])
    points1['t'] = np.array(points1['t'])
    return points0, points1


class Neuron:
    def __init__(self):
        self.psp_sum = 0.0  # 后突触电位的累加

    def integrate(self, psp):
        """
        集成所有突触的后突触电位。

        参数：
        - psp: 后突触电位的数组
        """
        self.psp_sum = np.sum(psp)

    def fire(self):
        """
        判断神经元是否发放动作电位（脉冲）。

        返回：
        - spike: 是否发放动作电位（True或False）
        """
        return self.psp_sum >= 1.0  # 一个简单的阈值判断


# 创建前突触神经元和后突触神经元
pre_synaptic_neuron = Neuron()
post_synaptic_neuron = Neuron()

# 创建一个突触，将前突触神经元连接到后突触神经元
synapse = Synapse(initial_weight=0.5)

# 模拟前突触神经元的活动
pre_synaptic_spike = 1

# 通过突触传递信号到后突触神经元
psp = synapse.transmit(pre_synaptic_spike)

# 集成后突触电位到后突触神经元
post_synaptic_neuron.integrate([psp])

# 判断后突触神经元是否发放脉冲
spike = post_synaptic_neuron.fire()

# 更新突触权重（根据可塑性规则）
delta_weight = 0.1  # 例如，根据STDP规则计算的值
synapse.update_weight(delta_weight)

print("后突触电位:", psp)
print("后突触神经元是否发放脉冲:", spike)
