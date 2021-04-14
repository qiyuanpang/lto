# -*- coding: utf-8 -*- 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import os
import pickle
import imp
import numpy as np
#sys.setdefaultencoding("utf-8")

sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
import gps
from gps import __file__ as gps_filepath
from gps.algorithm.policy_opt.lto_model import first_derivative_network
from gps.algorithm.policy.tf_policy import TfPolicy

def main():
    #cur_dir = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
    EXP_DIR = BASE_DIR + '/../experiments/laplace/'
    network_dir = EXP_DIR + 'data_files_pde/' + ('policy_itr_%02d' % 0) + '.pkl'
    hyperparams_file = EXP_DIR + 'hyperparams.py'
    hyperparams = imp.load_source('hyperparams', hyperparams_file).config['algorithm']
    #print(network_dir)
    #print(hyperparams.keys())
    pol_dict = pickle.load(open(network_dir, "rb"), encoding='latin1')
    print(pol_dict.keys(), pol_dict['scale'].shape, pol_dict['bias'].shape)
    #print(pol_dict['scale'])
    #print(pol_dict['bias'])
    network_config = hyperparams['policy_opt']['network_params']
    network_config['deg_action'] = 1050
    network_config['param_dim'] = network_config['deg_action']
    network_config['deg_obs'] = network_config['deg_action']*(network_config['history_len']*2 +1) + network_config['history_len']
    network = TfPolicy.load_policy(network_dir, first_derivative_network, network_config=network_config)
    np.random.seed(0)
    x = np.random.randn(network_config['deg_action'])
    np.random.seed(0)
    obs = np.random.randn(network_config['deg_obs'])
    act = network.act(x, obs, 0, None, usescale=False)
    print(x.shape, act.shape, obs.shape, act[0:20])

if __name__ == '__main__':
    main()
