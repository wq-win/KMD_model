import tempfile

import numpy as np
import copy
import matplotlib.pyplot as plt
import time
import dill
from matplotlib.backends.backend_pdf import PdfPages
from cycler import cycler
from collections import deque
import os
from tracereader import TraceReader
from tracelogger import TraceLogger
import matplotlib as mpl
class DynamicSynapseArray:
    def __init__(self, number_of_synapses=[1, 3], period=None, t_in_period=None, period_var=None,
                 amp=None, weighter_central=None, weighter_central_update_rate=0.000012,
                 weighter_oscillate_decay=0.0000003,
                 modulator_amount=0, init_amp=0.4, t=0, dt=1, normalized_weight=False, learning_rule_osci=True,
                 learning_rule_pre=False, name=''):

        self.number_of_synapses = number_of_synapses  # [1]
        self.dt = dt
        self.t = t
        self.period_centre = np.ones(number_of_synapses).astype(
            np.float64) * period if period is not None else 1000 + 100 * (np.random.rand(*number_of_synapses) - 0.5)
        self.period = copy.deepcopy(self.period_centre)
        self.t_in_period = np.ones(number_of_synapses).astype(
            np.float64) * t_in_period if t_in_period is not None else np.random.rand(*number_of_synapses) * self.period
        self.period_var = np.ones(number_of_synapses).astype(
            np.float64) * period_var if period_var is not None else np.ones(number_of_synapses).astype(np.float) * 0.1
        self.amp = np.multiply(np.ones(number_of_synapses).astype(np.float64), amp) if amp is not None else np.ones(
            number_of_synapses).astype(np.float) * 0.2
        self.weighters_central = np.multiply(np.ones(number_of_synapses),
                                             weighter_central) if weighter_central is not None else (np.random.randn(
            *number_of_synapses) - 0.5) * init_amp
        self.normalized_weight = normalized_weight
        print('self.weighters_central \n', self.weighters_central)
        sum_of_weighters_central = np.sum(self.weighters_central, axis=1)
        print('sum_of_weighters_central \n', sum_of_weighters_central)
        if normalized_weight:
            self.weighters_central /= np.sum(self.weighters_central, axis=1)[:, None]
        self.weighters_central_update_rate = np.ones(
            number_of_synapses) * weighter_central_update_rate if weighter_central_update_rate is not None else np.ones(
            number_of_synapses) * 0.000012
        self.weighters = self.weighters_central + self.amp * np.sin(self.t_in_period / self.period * 2 * np.pi)
        self.weighters_last = copy.deepcopy(self.weighters)
        self.weighters_oscilate_decay = np.ones(
            number_of_synapses) * weighter_oscillate_decay if weighter_oscillate_decay is not None else np.ones(
            number_of_synapses)
        self.modulator_amount = np.ones(number_of_synapses) * modulator_amount
        self.zero_cross = np.ones(number_of_synapses, dtype=bool)
        self.learning_rule_osci = learning_rule_osci
        self.learning_rule_pre = learning_rule_pre
        self.learning_rule_pre_factor = 1
        self.weighters_central_var = np.zeros(self.weighters_central.shape)

        if not name:
            self.name = 'DSA'
        else:
            self.name = name
        # if not trace_variable:
        #     self.trace_variable = ['t',
        #                            'weighters',
        #                            'weighters_central',
        #                            'period',
        #                            't_in_period',
        #                            'amp',
        #                            'modulator_amount',
        #                            ]

    def amp_control(self, rate=1):
        self.amp *= rate

    def step_synapse_dynamics(self, dt, t, modulator_amount, pre_syn_activity=0):
        self.weighters_central_var[:, :] = 0
        if dt is None:
            dt = self.dt
        self.t = t
        self.t_in_period += dt
        self.weighters = self.weighters_central + self.amp * np.sin(self.t_in_period / self.period * 2 * np.pi)
        modulator_amount_osci = modulator_amount
        modulator_amount_pre = 0.2

        if self.learning_rule_osci:
            self.weighters_central_var += (
                                                      self.weighters - self.weighters_central) * modulator_amount_osci * self.weighters_central_update_rate * dt
        if self.learning_rule_pre:
            self.weighters_central_var += (
                                                      self.learning_rule_pre_factor * pre_syn_activity - self.weighters_central) * modulator_amount_pre * self.weighters_central_update_rate * dt

        assert not np.any(np.isnan(self.weighters_central)), "weighter_central has nan" + str(
            np.any(np.isnan(self.weighters_central))) + "self.weighters_central_var=" + str(
            self.weighters_central_var) + "\nself.learning_rule_osci" + str(
            self.learning_rule_osci) + "\nself.learning_rule_pre" + str(
            self.learning_rule_pre) + "\npre_syn_activity" + str(pre_syn_activity) + "\nself.weighter_central" + str(
            self.weighters_central)
        assert not np.any(np.isnan(
            self.weighters_central_var)), "weighters_central_var has nan" + "self.weighters_central_var=" + str(
            self.weighters_central_var) + "\nself.learning_rule_osci" + str(
            self.learning_rule_osci) + "\nself.learning_rule_pre" + str(
            self.learning_rule_pre) + "\npre_syn_activity" + str(pre_syn_activity) + "\nself.weighter_central" + str(
            self.weighters_central)

        self.weighters_central += self.weighters_central_var

        if self.normalized_weight:
            self.weighters_central /= np.sum(np.abs(self.weighters_central), axis=1)[:, None]

        self.modulator_amount = np.ones(self.number_of_synapses) * modulator_amount
        self.amp *= np.exp(-self.weighters_oscilate_decay * self.modulator_amount * dt)
        self.zero_cross = np.logical_and(np.less(self.weighters_last, self.weighters_central),
                                         np.greater_equal(self.weighters, self.weighters_central))
        self.t_in_period[self.zero_cross] = self.t_in_period[self.zero_cross] % self.period[self.zero_cross]
        # todo IndexError: boolean index did not match indexed array along dimension 1; dimension is 1 but corresponding boolean dimension is 7

        #        self.period[self.zero_cross] += (np.random.rand(*self.number_of_synapses)[self.zero_cross]-0.5)*self.period_var[self.zero_cross]*self.period[self.zero_cross]+(self.period_centre-self.period)[self.zero_cross]*0.03
        self.period[self.zero_cross] = np.random.normal(loc=self.period_centre[self.zero_cross],
                                                        scale=self.period_centre[self.zero_cross] * 0.1)
        self.weighters_last = self.weighters
        return self.weighters

    def init_recording(self, name_list=[], log_path='', log_name=''):
        if not name_list:
            name_list = ['weighters',
                         'weighters_central',
                         'modulator_amount',
                         'amp',
                         'period',
                         't_in_period',
                         't']
        self.name_list = name_list
        if not log_name:
            log_name = self.name
        if not log_path:
            log_path = tempfile.gettempdir()
            self.log_file_path = os.path.join(log_path, log_name + '.pkl')
        else:
            if log_path.endswith('.pkl'):
                self.log_file_path = log_path
            else:
                self.log_file_path = os.path.join(log_path, log_name + '.pkl')
        if not os.path.exists(os.path.dirname(self.log_file_path)) and os.path.dirname(self.log_file_path):
            os.makedirs(os.path.dirname(self.log_file_path))
        self.trace_logger = TraceLogger(name_list, self.log_file_path)

    def recording(self):
        temp_dict = {}
        for item in self.name_list:
            # exec("temp = self.%s" % (key))
            temp_dict[item] = getattr(self, item)
        self.trace_logger.append(temp_dict)

    def memory_maintenance(self):
        self.trace_logger.memory_maintenance()

    def save_recording(self):
        self.trace_logger.save_trace()

    def clear_record_cache(self):
        self.trace_logger.clear_cache()

    def retrieve_record(self):
        self.trace_reader = TraceReader(self.log_file_path)
        self.trace = self.trace_reader.get_trace()
        return self.trace

    # %%
    def plot(self, path='', save_plots=False, start_time_rate=0, down_sample_rate=10, line_width=1, full_scale=False,
             name_str=None, neuron_id=0):
        #    plt.rc('axes', prop_cycle=(cycler('color',['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','b','k'])))
        mpl.rcParams['axes.prop_cycle'] = cycler('color',
                                                 ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                                                  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'b', 'k'])
        #    mpl.rcParams['axes.prop_cycle']=cycler(color='category20')
        #        if trace is None:
        #            trace = self.trace
        self.retrieve_record()
        if name_str is None:
            name_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        tracet_in_s = np.array(self.trace['t'])[::down_sample_rate].astype(float) / 1000
        number_of_steps = len(tracet_in_s)
        if start_time_rate == 0:
            start_step = 0
        else:
            start_step = int(number_of_steps * start_time_rate)

        figure_dict = {}
        figure_dict['ASynapse'] = plt.figure()
        synapse_id = 0
        figure0lines0, = plt.plot(tracet_in_s,
                                  np.array(self.trace['modulator_amount'])[::down_sample_rate, neuron_id, synapse_id],
                                  line_width=1)
        figure0lines1, = plt.plot(tracet_in_s,
                                  np.array(self.trace['weighters_central'])[::down_sample_rate, neuron_id, synapse_id],
                                  line_width=1)
        figure0lines2, = plt.plot(tracet_in_s,
                                  np.array(self.trace['weighters'])[::down_sample_rate, neuron_id, synapse_id],
                                  line_width=1)
        plt.legend([figure0lines2, figure0lines1, figure0lines0],
                   ['Weight Fluctuation', 'Fluctuation Centre', 'Modulator Amount', ], loc=4)
        plt.title('Example Dynamics of a Synapse')
        #        plt.ylim([-2,2])
        figure_dict['weighters'] = plt.figure()

        labels = [str(i) for i in range(self.trace['weighters'][0].shape[1])]
        figure1lines = plt.plot(tracet_in_s, np.array(self.trace['weighters'])[::down_sample_rate, neuron_id],
                                label=labels,
                                line_width=line_width)
        plt.legend(figure1lines, labels)
        plt.xlabel('Time (s)')
        plt.title('Instantaneous Synaptic Strength')

        trace_weighters = np.array(self.trace['weighters'])[start_step:][::down_sample_rate, :, :]
        if trace_weighters.shape[2] >= 3:

            X = trace_weighters[:, neuron_id, 0]
            Y = trace_weighters[:, neuron_id, 1]
            Z = trace_weighters[:, neuron_id, 2]
        else:
            trace_weighters = trace_weighters.reshape(trace_weighters.shape[0], -1)
            X = trace_weighters[:, 0]
            Y = trace_weighters[:, 1]
            if trace_weighters.shape[1] > 2:
                Z = trace_weighters[:, 2]
            else:
                Z = np.zeros_like(Y)

        figure_dict['2Weighters'] = plt.figure()
        plt.plot(X, Y)
        plt.xlabel('Time (s)')
        plt.title('2 Instantaneous Synaptic Strength')
        plt.xlabel('Instantaneous Synaptic Strength 0')
        plt.ylabel('Instantaneous Synaptic Strength 1')

        figure_dict['3Weighters'] = plt.figure()
        ax = figure_dict['3Weighters'].add_subplot(111, projection='3d')
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
            ax.plot([xb], [yb], [zb], 'w', line_width=line_width)
        if full_scale:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)

        figure_dict['weighters_central'] = plt.figure()
        figure4lines = plt.plot(tracet_in_s, np.array(self.trace['weighters_central'])[::down_sample_rate, neuron_id],
                                label=labels, line_width=line_width)
        plt.legend(figure4lines, labels)
        plt.title('Center of Synaptic Strength Oscillation')
        plt.xlabel('Time (s)')

        figure_dict['period'] = plt.figure()
        figure5lines = plt.plot(tracet_in_s, np.array(self.trace['period'])[::down_sample_rate, neuron_id],
                                label=labels,
                                line_width=line_width)
        plt.legend(figure5lines, labels)
        plt.title('period')
        plt.xlabel('Time (s)')
        plt.xlabel('period (s)')
        figure_dict['t_in_period'] = plt.figure()
        figure6lines = plt.plot(tracet_in_s, np.array(self.trace['t_in_period'])[::down_sample_rate, neuron_id],
                                label=labels,
                                line_width=line_width)
        plt.legend(figure6lines, labels)
        plt.title('t_in_period')
        plt.xlabel('Time (s)')
        plt.xlabel('period (s)')
        figure_dict['amp'] = plt.figure()
        figure6lines = plt.plot(tracet_in_s, np.array(self.trace['amp'])[::down_sample_rate, neuron_id], label=labels,
                                line_width=line_width)
        plt.legend(figure6lines, labels)
        plt.title('amp')
        plt.xlabel('Time (s)')

        figure_dict['modulator_amount'] = plt.figure()
        figure7lines = plt.plot(tracet_in_s, np.array(self.trace['modulator_amount'])[::down_sample_rate, neuron_id],
                                label=labels, line_width=line_width)
        plt.legend(figure7lines, labels)
        plt.xlabel('Time (s)')
        plt.title('modulator_amount')
        # %
        figure_dict['PoincareMap'] = plt.figure()
        figure8ax1 = figure_dict['PoincareMap'].add_subplot(111)

        trace_weighters = np.array(self.trace['weighters'])[::down_sample_rate, :, :]
        if trace_weighters.shape[2] >= 3:
            points0, points1 = cross_analysis(trace_weighters[:, neuron_id, 0],
                                              np.array(self.trace['weighters_central'])[::down_sample_rate, neuron_id,
                                              0],
                                              trace_weighters[:, neuron_id], tracet_in_s)
        else:
            trace_weighters = trace_weighters.reshape(trace_weighters.shape[0], -1)
            points0, points1 = cross_analysis(trace_weighters[:, 0],
                                              np.array(self.trace['weighters_central'])[::down_sample_rate, neuron_id,
                                              0],
                                              trace_weighters, tracet_in_s)
        if full_scale:
            figure8ax1.set_xlim(0, 1)
            figure8ax1.set_ylim(0, 1)
        print('points0')
        print(points0['points'])
        print('points1')
        print(points1['points'])

        pointsploted0 = figure8ax1.scatter(points0['points'][:, 0], points0['points'][:, 1], c=points0['t'],
                                           cmap=mpl.colormaps.get_cmap('Greens'), marker=".",
                                           edgecolor='none')  # c=c, ,  cmap=cm
        pointsploted1 = figure8ax1.scatter(points1['points'][:, 0], points1['points'][:, 1], c=points1['t'],
                                           cmap=mpl.colormaps.get_cmap('Blues'), marker=".", edgecolor='none')
        # plt.legend(figure7lines, labels)
        plt.colorbar(pointsploted0)
        plt.colorbar(pointsploted1)
        plt.title('Poincare map')
        plt.xlabel('Instantaneous Synaptic Strength 1')
        plt.ylabel('Instantaneous Synaptic Strength 2')
        # %
        #        for key in figure_dict:
        #            figure_dict[key].tight_layout()

        if save_plots:
            if path is None or not path:
                path = tempfile.tempdir(suffix=tim_of_recording, prefix='DS')
            if not os.path.exists(path):
                os.makedirs(path)
            pp = PdfPages(path + "DynamicSynapse" + name_str + '.pdf')
            for key in figure_dict:
                figure_dict[key].savefig(pp, format='pdf')
            pp.close()
        #        Figures = {'TraceWeighters':figure1, 'TraceWeighterVarRates':figure2, 'TraceWeighterInAxon':figure3, '2TraceWeighters':figure4, '3DTraceWeighters':figure5, 'weighters_central':figure6, 'Damping':figure7,'EquivalentVolume':figure8,'Poincare map':figure9}
        #        with open(path+"DynamicSynapse"+tim_of_recording+'.pkl', 'wb') as pkl:
        #            dill.dump(Figures, pkl)

        return figure_dict, ax

    # %%


def cross_analysis(oscillate, reference, oscillate_array, trace_t):
    points0 = {'t': [], 'points': []}
    points1 = {'t': [], 'points': []}
    greater_than_centre = (oscillate[0] > reference[0])
    print(oscillate[0])
    print(reference[0])
    for i1 in range(len(oscillate)):

        #        print(Oscillates[i1,0])
        #        print(References[i1,0])
        if greater_than_centre:
            if oscillate[i1] < reference[i1]:
                # print(greater_than_centre)
                # print(Oscillates[i1,0])
                points0['points'].append(oscillate_array[i1])
                points0['t'].append(trace_t[i1])
                greater_than_centre = False
        elif not greater_than_centre:
            if oscillate[i1] > reference[i1]:
                # print (greater_than_centre)
                # print(Oscillates[i1,0])
                points1['points'].append(oscillate_array[i1])
                points1['t'].append(trace_t[i1])
                greater_than_centre = True
    # c = np.empty(len(m[:,0])); c.fill(megno)
    points0['points'] = np.array(points0['points'])
    points1['points'] = np.array(points1['points'])
    points0['t'] = np.array(points0['t'])
    points1['t'] = np.array(points1['t'])
    return points0, points1

    # adsra: a dynamic synapse receptor amount


def simulation_loop(adsra, dt, number_of_steps, arg0, arg1, phase=0, index0=0, index1=0):
    adsra.tauWV = arg0
    adsra.aWV = arg1

    adsra.init_recording(number_of_steps)
    trace_t = np.zeros(number_of_steps)
    for step_index in range(number_of_steps):
        #        weighters_last = copy.deepcopy(weighters)
        #        WeighterVarRatesLast = copy.deepcopy(WeighterVarRates)
        adsra.StateUpdate()
        adsra.step_synapse_dynamics(simulation_time_interval, 0)
        if adsra.RecordingState:
            adsra.recording()
            trace_t[step_index] = step_index * simulation_time_interval
            # %%
        if step_index % (100000. / dt) < 1:
            print('phase=%s,index0=%d, index1=%d, tauWV=%s, aWV=%s, step_index=%s' % (
                phase, index0, index1, adsra.tauWV, adsra.aWV, step_index))

    return adsra


if __name__ == "__main__":
    inintial_ds = 1
    single_simulation = 1
    tim_of_recording = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    if inintial_ds:
        simulation_time_lenth = 10 * 60 * 1000
        period = 20000
        dt = simulation_time_interval = 33
        number_of_steps = int(simulation_time_lenth / simulation_time_interval)
        period_steps = int(period / simulation_time_interval)
        modulator_amount = np.zeros(number_of_steps)
        modulator_amount[int(period_steps / 2):period_steps] = 1
        modulator_amount[period_steps * 2:int(period_steps * 2 + period_steps / 2)] = 1
        number_of_neuron = 2
        number_of_synapses = 3
        weighters_central = 0  # np.ones((number_of_neuron,number_of_synapses))*0+ 0.4* (np.random.rand(number_of_neuron,number_of_synapses)-0.5)

        adsra = DynamicSynapseArray(number_of_synapses=(number_of_neuron, number_of_synapses), period=period,
                                    t_in_period=None, period_var=0.1,
                                    amp=1, weighter_central=weighters_central, weighter_central_update_rate=0.000012,
                                    weighter_oscillate_decay=0.0000003)  # t_in_period=None

        adsra.init_recording()
    # %%
    if single_simulation:
        # %%
        for step in range(number_of_steps):

            adsra.step_synapse_dynamics(simulation_time_interval, step * simulation_time_interval, 0)
            adsra.recording()
            if step % 1000 == 0:
                print('%d of %d steps' % (step, number_of_steps))
        adsra.save_recording()
        # Please replace path with your current path + /Plots/. You can use os.getcwd() to get it.
        FigureDict, ax = adsra.plot(
            path='E:\\PycharmProjects\\DynamicSynapseShared\\Plots\\', down_sample_rate=100,
            save_plots=False, line_width=0.2, name_str=tim_of_recording)  # path=
        # %%
        plt.show()