import os
import numpy as np
import time
import uuid
import Synapse as DSA
import matplotlib.pyplot as plt


class DynamicSynapseArrayDict:
    def __init__(self, dsa_list, name_list=None, name='DSADict'):
        if name_list is None:
            self.name_list = list()
            for i1 in range(len(dsa_list)):
                self.name_list.append(str(uuid.uuid1().hex))
        else:
            self.name_list = name_list
        self.name = name
        self.dsa_dict = dict()
        for i1 in range(len(dsa_list)):
            self.dsa_dict[self.name_list[i1]] = dsa_list[i1]
        self.linked_dsas = {}

    def amp_control(self, rate=1):
        for key in self.dsa_dict:
            self.dsa_dict[key].amp_control(rate)
        if self.linked_dsas:
            for key in self.linked_dsas:
                self.linked_dsas[key].amp_control()

    def link_multiple_dsas(self, dsas_list, name_list=None):
        for i1 in range(len(dsas_list)):
            self.link_dsas(dsas_list[i1], name_list[i1])

    def link_dasa(self, dsas, name=None):
        if name is None:
            name = str(uuid.uuid1().hex)
        self.linked_dsas[name] = dsas

    def step_synapse_dynamics(self, dt, t, modulator_amount, pre_syn_activity=0):
        for key in self.dsa_dict:
            self.dsa_dict[key].step_synapse_dynamics(dt, t, modulator_amount, pre_syn_activity=pre_syn_activity)
        if self.linked_dsas:
            for key in self.linked_dsas:
                self.linked_dsas[key].step_synapse_dynamics(dt, t, modulator_amount, pre_syn_activity=pre_syn_activity)

    def init_recording(self, log_path=''):
        self.log_path = os.path.join(log_path, self.name)
        if not os.path.exists(os.path.dirname(self.log_path)) and os.path.dirname(self.log_path):
            os.makedirs(os.path.dirname(self.log_path))
        for key in self.dsa_dict:
            self.dsa_dict[key].init_recording(log_path=os.path.join(self.log_path, key))
        if self.linked_dsas:
            for key in self.linked_dsas:
                self.linked_dsas[key].init_recording(log_path=os.path.join(self.log_path, 'linked_dsas', key))

    def recording(self):
        for key in self.dsa_dict:
            self.dsa_dict[key].recording()
        if self.linked_dsas:
            for key in self.linked_dsas:
                self.linked_dsas[key].recording()

    def clear_record_cache(self):
        for key in self.dsa_dict:
            self.dsa_dict[key].clear_record_cache()
        if self.linked_dsas:
            for key in self.linked_dsas:
                self.linked_dsas[key].clear_record_cache()

    def save_recording(self):
        for key in self.dsa_dict:
            self.dsa_dict[key].save_recording()
        if self.linked_dsas:
            for key in self.linked_dsas:
                self.linked_dsas[key].save_recording()

    # %%
    def plot(self, path='', save_plots=False, start_time_rate=0, down_sample_rate=10, linewidth=1, full_scale=False,
             name_str=None, neuron_id=0, recurrently=False):
        figure_dict_dict = dict()
        ax_dict = dict()
        for key in self.dsa_dict:
            figure_dict_dict[key], ax_dict[key] = self.dsa_dict[key].plot(path=path + key, save_plots=save_plots,
                                                                          start_time_rate=start_time_rate,
                                                                          down_sample_rate=down_sample_rate,
                                                                          linewidth=linewidth, full_scale=full_scale,
                                                                          name_str=name_str, neuron_id=neuron_id)
        if recurrently:
            if self.linked_dsas:
                for key in self.linked_dsas:
                    self.linked_dsas[key].plot(path=path + key, save_plots=save_plots, start_time_rate=start_time_rate,
                                               down_sample_rate=down_sample_rate, linewidth=linewidth,
                                               full_scale=full_scale,
                                               name_str=name_str, neuron_id=neuron_id)
        return figure_dict_dict, ax_dict


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
        modulator_amount[int(period_steps) * 2:int(period_steps) * 2 + int(period_steps / 2)] = 1
        number_of_neuron = 2
        number_of_synapses = 3
        weighters_central = 0   # np.ones((number_of_neuron,number_of_synapses))*0+ 0.4* (np.random.rand(number_of_neuron,number_of_synapses)-0.5)

        dsra0 = DSA.DynamicSynapseArray(number_of_synapses=(number_of_neuron, number_of_synapses), period=period,
                                        t_in_period=None, period_var=0.1,
                                        amp=1, weighter_central=weighters_central,
                                        weighter_central_update_rate=0.000012,
                                        weighter_oscillate_decay=0.0000003)  # t_in_period=None
        dsra1 = DSA.DynamicSynapseArray(number_of_synapses=(number_of_neuron, number_of_synapses), period=period,
                                        t_in_period=None, period_var=0.1,
                                        amp=1, weighter_central=weighters_central,
                                        weighter_central_update_rate=0.000012,
                                        weighter_oscillate_decay=0.0000003)  # t_in_period=None
        dsra2 = DSA.DynamicSynapseArray(number_of_synapses=(number_of_neuron, number_of_synapses), period=period,
                                        t_in_period=None, period_var=0.1,
                                        amp=1, weighter_central=weighters_central,
                                        weighter_central_update_rate=0.000012,
                                        weighter_oscillate_decay=0.0000003)  # t_in_period=None
        adsra_dict = DynamicSynapseArrayDict([dsra0, dsra1, dsra2])
        adsra_dict.init_recording()
    # %%
    if single_simulation:
        # %%
        for step in range(number_of_steps):

            adsra_dict.step_synapse_dynamics(simulation_time_interval, step * simulation_time_interval, 0)
            adsra_dict.recording()
            if step % 1000 == 0:
                print('%d of %d steps' % (step, number_of_steps))
        adsra_dict.save_recording()
        path = os.getcwd()
        figure_dict_dict, ax_dict = adsra_dict.plot(path=path + '\\log\\SynapseDict\\', down_sample_rate=1,
                                                    save_plots=True, linewidth=0.2, name_str=tim_of_recording)  # path=
    # %%
    plt.show()
