""" This file defines the main object that runs experiments. """
# Difference from gps_main.py: Uses a workaround to save Tensorflow policy. 

import logging
import imp
import os
import os.path
import sys
import argparse
import time
import numpy as np
import random
import cProfile
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#logging.basicConfig(level=logging.DEBUG)
# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
import gps as gps_globals
from gps.utility.display import Display
from gps.sample.sample_list import SampleList
from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.lto_model import first_derivative_network, first_derivative_network_leaky_relu, first_derivative_network_swish
from gps.agent.lto.agent_lto import AgentLTO
#os.environ['CUDA_VISIBLE_DEVICES']='0, 1'

class GPSMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config):
        """
        Initialize GPSMain
        Args:
            config: Hyperparameters for experiment
        """
        self._hyperparams = config
        self._conditions = config['common']['conditions']
        if 'train_conditions' in config['common']:
            self._train_idx = config['common']['train_conditions']
            self._test_idx = config['common']['test_conditions']
        else:
            self._train_idx = range(self._conditions)
            config['common']['train_conditions'] = config['common']['conditions']
            self._hyperparams=config
            self._test_idx = self._train_idx

        self._data_files_dir = config['common']['data_files_dir']

        self.agent = config['agent']['type'](config['agent'])
        self.disp = Display(config['common'])     # For logging
        
        config['algorithm']['agent'] = self.agent
        self.algorithm = config['algorithm']['type'](config['algorithm'])

    def run(self, time):
        
        itr_start = 0
        
        for itr in range(itr_start, self._hyperparams['iterations']):
            for m, cond in enumerate(self._train_idx):
                for i in range(self._hyperparams['num_samples']):
                    self._take_sample(itr, cond, m, i)

            traj_sample_lists = [self.agent.get_samples(cond, -self._hyperparams['num_samples']) for cond in self._train_idx]
### what is -self._hyperparams['num_samples'] for ??
            # Clear agent samples.
            self.agent.clear_samples()
            print(traj_sample_lists)
            
            self.algorithm.iteration(traj_sample_lists)
            
            pol_sample_lists = self._take_policy_samples(self._train_idx)
            
            self._prev_traj_costs, self._prev_pol_costs = self.disp.update(itr, self.algorithm, self.agent, traj_sample_lists, pol_sample_lists)
            self.algorithm.policy_opt.policy.pickle_policy(self.algorithm.policy_opt._dO, self.algorithm.policy_opt._dU, self._data_files_dir + ('policy_itr_%02d' % (itr+time*self._hyperparams['iterations'])))
        
    def destroy(self):
        if 'on_exit' in self._hyperparams:
            self._hyperparams['on_exit'](self._hyperparams)
        #return cs
    
    def _take_sample(self, itr, cond, m, i):
        
        if self.algorithm.iteration_count == 0:
            pol = self.algorithm.cur[m].traj_distr
        else:
            if self.algorithm._hyperparams['sample_on_policy']:
                pol = self.algorithm.policy_opt.policy
            else:
                pol = self.algorithm.cur[m].traj_distr
        
        self.agent.sample(pol, cond)

    def _take_policy_samples(self, cond_list):
        pol_samples = [[] for _ in range(len(cond_list))]
        for cond in range(len(cond_list)):
            for i in range(self._hyperparams['num_samples']):
                pol_samples[cond].append(self.agent.sample(self.algorithm.policy_opt.policy, cond_list[cond], save=False))
        return [SampleList(samples) for samples in pol_samples]

def updateconfig(config, param_dim, history_len):
    config['agent']['sensor_dims'][CUR_LOC] = param_dim
    config['agent']['sensor_dims'][PAST_OBJ_VAL_DELTAS] = history_len
    config['agent']['sensor_dims'][PAST_GRADS] = history_len*param_dim
    config['agent']['sensor_dims'][PAST_LOC_DELTAS] = history_len*param_dim
    config['agent']['sensor_dims'][CUR_GRAD] = param_dim
    config['agent']['sensor_dims'][ACTION] = param_dim

    config['agent']['state_include'] = [config['agent']['sensor_dims'][CUR_LOC]]
    config['agent']['obs_include'] = [config['agent']['sensor_dims'][PAST_OBJ_VAL_DELTAS], config['agent']['sensor_dims'][PAST_GRADS], config['agent']['sensor_dims'][CUR_GRAD], config['agent']['sensor_dims'][PAST_LOC_DELTAS]]
    config['agent']['history_len'] = history_len

    config['algorithm']['policy_opt']['network_params']['param_dim'] = param_dim
    config['algorithm']['policy_opt']['network_params']['history_len'] = history_len

def Train(exp_dir, config, times):
    
    for i in range(times):
        print('************************************************************************************')
        print('******************** The ' + '%02d' % i + ' training starts *************************')
        #print(config['agent']['fcns'][0]['init_loc'])
        if config['common'].get('train_conditions'):
            del config['common']['train_conditions']
        if i == 0:
            gps = GPSMain(config)
            gps.run(i)
            gps.destroy()
            del gps
        else:
            #gps = GPSMain(config)
            dim = np.random.randint(12, 17, size=1)[0]
            session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True))
            fcns, fcn_family = config['agent']['gen_fcns'](session, dim, dim)
            config['agent']['fcns'] = fcns
            config['agent']['fcn_family'] = fcn_family
            updateconfig(config, fcns[0]['dim'], config['agent']['history_len'])
            for k in range(i):
                Agent = AgentLTO(config['agent'])
                network_dir = exp_dir + 'data_files_pde/' + ('policy_itr_%02d' % (k*config['iterations']+1)) + '.pkl'
                lr_pol = TfPolicy.load_policy(network_dir, 1, first_derivative_network, network_config=config['algorithm']['policy_opt']['network_params'])
                for j in range(config['common']['conditions']):
                    config['agent']['fcns'][j]['init_loc'] = np.expand_dims(Agent.sample(lr_pol, j, verbose=False, save=False, noisy=False, usescale=False).get_X()[-1], axis=1) 
            config['algorithm']['policy_opt']['policy_dict_path'] = network_dir
            tf.reset_default_graph()
            gps = GPSMain(config)
            gps.run(i)
            gps.destroy()
            del gps
        print(config['agent']['fcns'][0]['init_loc'])           
        print('******************************* This training ends ****************************************')
        
def main():
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str, help='experiment name')
    args = parser.parse_args()

    exp_name = args.experiment

    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'
    #print(hyperparams_file)

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" % (exp_name, hyperparams_file))
    
    # May be used by hyperparams.py to load different conditions
    gps_globals.phase = "TRAIN"
    hyperparams = imp.load_source('hyperparams', hyperparams_file)

    seed = hyperparams.config.get('random_seed', 0)
    random.seed(seed)
    np.random.seed(seed)
    #print('====================================')
    #print(hyperparams.config)
    #print('====================================')
    
    gpu_ids = hyperparams.config['algorithm']['policy_opt']['gpu_ids']
    gpus = str(gpu_ids)[1:-1]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    

    #gps = GPSMain(hyperparams.config)
    #cProfile.run(gps.run())
    #gps.run()
    
    times = 5
    Train(exp_dir, hyperparams.config, times)
        
    if 'on_exit' in hyperparams.config:
        hyperparams.config['on_exit'](hyperparams.config)


if __name__ == "__main__":
    main()
    #cProfile.run('main()')
