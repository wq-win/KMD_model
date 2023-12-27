import sys
import Synapse as DSA
import SynapseDict as DSADict
import os
import Utils.RangeAdapter as RA
import numpy as np
import matplotlib.pyplot as plt


class DenseLayer:

    def __init__(self, number_of_input, number_of_neuron, act_fun=np.tanh, adaptation=False, a=None, b=None, c=None,
                 name='denselayer'):
        self.number_of_input = number_of_input
        self.number_of_neuron = number_of_neuron
        self.act_fun = act_fun
        self.neuron_activity = np.zeros((self.number_of_neuron, 1))
        self.adaptation = adaptation
        if a is None:
            self.a = np.random.randn(self.number_of_neuron, self.number_of_input)
        elif np.isscalar(a):
            self.a = np.full((self.number_of_neuron, self.number_of_input), a)
        else:
            assert np.array(a).shape == (self.number_of_neuron, self.number_of_input), "a has a wrong shape"
            self.a = a
        if b is None:
            self.b = np.random.randn(self.number_of_neuron, 1)
        elif np.isscalar(b):
            self.b = np.full((self.number_of_neuron, 1), b)
        else:
            assert np.array(b).shape == (self.number_of_neuron, 1), "a has a wrong shape"
            self.b = b
        if c is None:
            self.c = np.random.randn(self.number_of_neuron, 1)
        elif np.isscalar(c):
            self.c = np.full((self.number_of_neuron, 1), c)
        else:
            assert np.array(c).shape == (self.number_of_neuron, 1), "a has a wrong shape"
            self.c = c
        self.adjustable_params_name = ('a', 'b', 'c')
        self.name = name

    def assign_range_adapter(self, targe_max_output=0.5, targe_min_output=-0.5, factor=1, bias=0,
                             update_rate=0.00005, t=0):
        self.adaptation = True
        self.range_adapter = RA.RangeAdapter(targe_max_output=targe_max_output, targe_min_output=targe_min_output,
                                             factor=factor, bias=bias, update_rate=update_rate, t=t)

    def assign_dsa(self, period=20000, t_in_period=None, period_var=0.1,
                   amp=0.2, weighters_central=None, weighters_central_update_rate=0.000012,
                   weighters_oscillate_decay=0.0000003 / 100, normalized_weight=False):
        # self.ADSA = DSA.DynamicSynapseArray((self.number_of_neuron, self.number_of_input[0]*self.number_of_input[1]),
        #                                     period=period,
        #                                     t_in_period=t_in_period, period_var=period_var,
        #                                     amp=amp, weighter_central=weighters_central,
        #                                     weighter_central_update_rate=weighters_central_update_rate,
        #                                     weighter_oscillate_decay=weighters_oscillate_decay,
        #                                     normalized_weight=normalized_weight)
        # return self.ADSA
        if weighters_central is None:
            weighters_central = {}
            weighters_central['a'] = self.a
            weighters_central['b'] = self.b
            weighters_central['c'] = self.c
        dsa_list = []
        dsa_list.append(
            DSA.DynamicSynapseArray((self.number_of_neuron, self.number_of_input),
                                    period=period,
                                    t_in_period=t_in_period, period_var=period_var,
                                    amp=amp, weighter_central=weighters_central['a'],
                                    weighter_central_update_rate=weighters_central_update_rate,
                                    weighter_oscillate_decay=weighters_oscillate_decay,
                                    normalized_weight=normalized_weight))
        dsa_list.append(DSA.DynamicSynapseArray((self.number_of_neuron, 1), period=period,
                                                t_in_period=t_in_period, period_var=period_var,
                                                amp=amp, weighter_central=weighters_central['b'],
                                                weighter_central_update_rate=weighters_central_update_rate,
                                                weighter_oscillate_decay=weighters_oscillate_decay,
                                                normalized_weight=normalized_weight))
        dsa_list.append(DSA.DynamicSynapseArray((self.number_of_neuron, 1), period=period,
                                                t_in_period=t_in_period, period_var=period_var,
                                                amp=amp, weighter_central=weighters_central['c'],
                                                weighter_central_update_rate=weighters_central_update_rate,
                                                weighter_oscillate_decay=weighters_oscillate_decay,
                                                normalized_weight=normalized_weight))

        self.dsas = DSADict.DynamicSynapseArrayDict(dsa_list, ['a', 'b', 'c'])
        return self.dsas

    def amp_control(self, rate=1):
        self.dsas.amp_control(rate)

    def step(self, input_value, dt=20):
        post_synaptic = np.matmul(self.a, input_value)
        self.neuron_inner_activity = post_synaptic + self.b.ravel()
        #        print(self.ADSA.weighters)
        if self.adaptation:
            self.neuron_inner_activity_adapted = self.range_adapter.step_dynamics(dt, self.neuron_inner_activity)
            self.neuron_activity = self.act_fun(self.neuron_inner_activity_adapted) + self.c.ravel()
        else:
            self.neuron_activity = self.act_fun(self.neuron_inner_activity) + self.c.ravel()

        return self.neuron_activity

    def update(self):
        if hasattr(self, 'range_adapter'):
            self.range_adapter.update()

    def step_synapse_dynamics(self, dt, t, modulator_amount, pre_syn_activity=0):
        self.dsas.step_synapse_dynamics(dt, t, modulator_amount, pre_syn_activity=pre_syn_activity)
        self.a = self.dsas.dsa_dict['a'].weighters  # reshape(self.number_of_neuron, self.number_of_input)
        self.b = self.dsas.dsa_dict['b'].weighters
        self.c = self.dsas.dsa_dict['c'].weighters

    def init_recording(self, log_path=''):
        self.log_path = os.path.join(log_path, self.name)
        if not os.path.exists(os.path.dirname(self.log_path)) and os.path.dirname(self.log_path):
            os.makedirs(os.path.dirname(self.log_path))
        self.dsas.init_recording(log_path=self.log_path)
        if hasattr(self, 'range_adapter'):
            self.range_adapter.init_recording(log_path=self.log_path)

    def recording(self):
        self.dsas.recording()
        if hasattr(self, 'range_adapter'):
            self.range_adapter.recording()

    def save_recording(self):
        self.dsas.save_recording()
        if hasattr(self, 'range_adapter'):
            self.range_adapter.save_recording()

    def memory_maintenance(self):
        self.dsas.memory_maintenance()
        if hasattr(self, 'range_adapter'):
            self.range_adapter.memory_maintenance()

    def clear_record_cache(self):
        self.dsas.clear_record_cache()
        if hasattr(self, 'range_adapter'):
            self.range_adapter.clear_record_cache()

    def plot(self, path='', save_plots=False, start_time_rate=0.3, down_sample_rate=10, linewidth=1, full_scale=False,
             name_str=None, neuron_id=0):
        self.dsas.plot(path=path, save_plots=save_plots, start_time_rate=start_time_rate,
                       down_sample_rate=down_sample_rate, linewidth=linewidth, full_scale=full_scale,
                       name_str=name_str, neuron_id=neuron_id)
        if self.adaptation:
            self.range_adapter.plot(path=path, save_plots=save_plots, start_time_rate=start_time_rate,
                                    neuron_id=neuron_id, down_sample_rate=down_sample_rate)


if __name__ == "__main__":
    Input = np.random.rand(7)
    dsl = DenseLayer(7, 3)
    path = os.getcwd()
    if sys.platform in ['darwin','linux']:
        path += '/log/'
    else:
        path += '\\log\\'
    dsl.assign_dsa()
    dsl.init_recording(log_path=path)

    T = 0
    dt = 33
    for i1 in range(100000):
        T += dt
        modulator_amount = 1
        dsl.step_synapse_dynamics(dt, T, modulator_amount)
        Output1 = dsl.step(Input)
        dsl.recording()

        # print('Output1', Output1)
    dsl.save_recording()
    dsl.plot(path=path)
    plt.show()