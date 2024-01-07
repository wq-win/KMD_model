import collections
import copy
import dill
import os
import numpy as np
from DenseLayer import DenseLayer

  
def sigmoid(x):  
    return 1 / (1 + np.exp(-x))


class KMDNN:
    def __init__(self, name='kmdnn'):
        self.layers = collections.OrderedDict()

        self.layers['K_M'] = DenseLayer(6,3,sigmoid)
        self.layers['K_D'] = DenseLayer(6,3,sigmoid)
        self.layers['D_K'] = DenseLayer(3,6,sigmoid)
        self.name = name
        self.output_dict = {}
        self.input_dict = {'K_M':np.random.normal(0,0.1,6),
                           'K_D':np.random.normal(0,0.1,6),
                           'D_K':np.random.normal(0,0.1,3)}

    def amp_control(self, rate=1):
        for key in self.layers:
            self.layers[key].amp_control(rate)

    def assign_dsa(self):
        for key in self.layers:
            self.layers[key].assign_dsa()
    
    def inin_recording(self, log_path=''):
        pathdir = os.path.dirname(log_path)
        self.log_path = os.path.join(pathdir, self.name)
        if not os.path.exists(os.path.dirname(self.log_path)):
            os.makedirs(os.path.dirname(self.log_path))
        for key in self.layers:
            self.layers[key].init_recording(log_path=self.log_path)

    def recording(self):
        for key in self.layers:
            self.layers[key].recording()
    
    def clear_record_cache(self):
        for key in self.layers:
            self.layers[key].clear_record_cache()

    def save_recording(self):
        for key in self.layers:
            self.layers[key].save_recording()    

    def step(self, dt):
        self.layers['K_M'].step(self.input_dict['K_M'],dt)
        self.layers['K_D'].step(self.input_dict['K_D'],dt)
        self.layers['D_K'].step(self.input_dict['D_K'],dt)
        
    def update(self):
        for key in self.layers:
            self.layers[key].update()

    def step_synapse_dynamics(self, dt, t, modulator_amount, pre_syn_activity=0):
        for key in self.layers:
            self.layers[key].step_synapse_dynamics(dt, t, modulator_amount)

    def plot(self, path='',key='K_M'):
        self.layers['K_M'].plot(path)
    
    def save_model(self, save_path=''):
        if not save_path:
            self.save_path = os.path.join(self.log_path, self.name + 'model.pkl')
        elif save_path.endswith('.pkl'):
            self.save_path = save_path
        else:
            self.save_path = os.path.join(save_path, self.name + 'model.pkl')
        selfcopy = copy.deepcopy(self)
        for key1 in selfcopy.layers.keys():
            selfcopy.layers[key1].dsas.clear_record_cache()
        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))
        with open(self.save_path, 'ab') as fTraces:
            dill.dump(selfcopy, fTraces, protocol=dill.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    kmd = KMDNN()