import logging
import imp
import os
import os.path
import sys
import argparse
import time
import numpy as np
import random
import copy
#import Queue as queue
from collections import deque
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import json

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
import gps as gps_globals
from gps import __file__ as gps_filepath
from gps.utility.display import Display
from gps.sample.sample_list import SampleList
from gps.agent.lto.agent_lto import AgentLTO
from gps.algorithm.policy.lto.gd_policy import GradientDescentPolicy
from gps.algorithm.policy.lto.cg_policy import ConjugateGradientPolicy
from gps.algorithm.policy.lto.lbfgs_policy import LBFGSPolicy
from gps.algorithm.policy.lto.momentum_policy import MomentumPolicy
from gps.proto.gps_pb2 import CUR_LOC, PAST_OBJ_VAL_DELTAS, PAST_GRADS, CUR_GRAD, PAST_LOC_DELTAS, ACTION
from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.lto_model import fully_connected_tf_network
from gps.agent.lto.fcn import QuadraticNormFcnFamily, QuadraticNormFcn
from gps.algorithm.cost.cost_utils import RAMP_CONSTANT
from gps.elescatter import generate_Z, HSSBF_Zfun, findeq, findge
from gps.algorithm.policy_opt.lto_model import first_derivative_network, first_derivative_network_leaky_relu, first_derivative_network_swish

def f_fcn(x, y, k):
    return x*(y+1.0/k)

def g_fcn(x, y, k):
    return np.log(np.square(1+x+1.0/k) + np.square(y))

def def_cost(dx, dy, x_pts, y_pts, input_dim_1, input_dim_2, k, session):

    input_dim = (input_dim_1-2)*(input_dim_2-2)
    input_dim1 = input_dim_1*input_dim_2
    fcn_family = QuadraticNormFcnFamily(input_dim, gpu_id = 0, session = session)

    param_dim = fcn_family.get_total_num_dim()

    fcn_objs = []
    dx2 = 1/(dx*dx)
    dy2 = 1/(dy*dy)
    mx = np.max([dx2, dy2, 2*(dx2+dy2)])
    ind1 = []
    ind2 = []
    data = np.zeros([input_dim1, input_dim1])
    labels = np.zeros([input_dim1,1])
    for j in range(0, input_dim_2):
        for i in range(0, input_dim_1):
            if i == 0 or j == input_dim_2 - 1 or i == input_dim_1 - 1 or j == 0:
                ind2.append(j*input_dim_1+i)
                data[j*input_dim_1 + i, j*input_dim_1 + i] = 1.0
                labels[j*input_dim_1 + i] = g_fcn(x_pts[i], y_pts[j], k)
            else:
                ind1.append(j*input_dim_1+i)
                data[j*input_dim_1 + i, j*input_dim_1 + i - 1] = dx2
                data[j*input_dim_1 + i, j*input_dim_1 + i + 1] = dx2
                data[j*input_dim_1 + i, (j-1)*input_dim_1 + i] = dy2
                data[j*input_dim_1 + i, (j+1)*input_dim_1 + i] = dy2
                data[j*input_dim_1 + i, j*input_dim_1 + i] = -2*(dx2 + dy2)
                labels[j*input_dim_1 + i] = f_fcn(x_pts[i], y_pts[j], k)

    data1 = data[ind1,:][:,ind1]
    data2 = data[ind1,:][:,ind2]
    labels1 = labels[ind1,:]-np.matmul(data2,labels[ind2,:])

    data1 = data1/mx
    labels1 = labels1/mx
    U = []
    fcn = QuadraticNormFcn(fcn_family, data1, labels1, disable_subsampling = True)

    return fcn, fcn_family, labels1

def pde_policy_comp(hyperparams, exp_dir, pols, input_dim_1 = 128, input_dim_2 = 128, a = 1.0, b = 1.0, learning_rate = 0.0001, mem_len = 10, momentum = 0.9, err = 1e-4):
    dx = a/input_dim_1
    dy = b/input_dim_2
    x_pts = np.arange(0,a+dx/2,dx)
    y_pts = np.arange(0,b+dy/2,dy)
    k = np.random.rand(1)[0]+0.5
    #session = tf.Session()
    session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True))
    
    cost, fcn_family, b = def_cost(dx, dy, x_pts, y_pts, input_dim_1, input_dim_2, k, session)
    normb = np.linalg.norm(b)
    
    obs_flag = hyperparams.config['algorithm']['policy_opt']['network_params']

    history_len = hyperparams.config['agent']['history_len']
    param_dim = fcn_family.get_total_num_dim()
    #print(param_dim)
    init_loc = np.random.rand(param_dim, 1)
    fcns = [{'fcn_obj': cost, 'dim': param_dim, 'init_loc': init_loc}]

    SENSOR_DIMS = { 
        CUR_LOC: param_dim,
        PAST_OBJ_VAL_DELTAS: history_len,
        PAST_GRADS: history_len*param_dim,
        PAST_LOC_DELTAS: history_len*param_dim,
        CUR_GRAD: param_dim, 
        ACTION: param_dim
    }
    obs_flag['sensor_dims'] = SENSOR_DIMS
    obs_flag['param_dim'] = param_dim

    agent = {
            'substeps': hyperparams.config['agent']['substeps'],
            'conditions': 1,
            'dt': hyperparams.config['agent']['dt'],
            'T': hyperparams.config['agent']['T'],
            'sensor_dims': SENSOR_DIMS,
            'state_include': hyperparams.config['agent']['state_include'],
            'obs_include': hyperparams.config['agent']['obs_include'],
            'history_len': history_len,
            'fcns': fcns,
            'fcn_family': fcn_family
    }
    
    network_config = hyperparams.config['algorithm']['policy_opt']['network_params']
    network_config['deg_action'] = param_dim
    network_config['param_dim'] = network_config['deg_action']
    network_config['deg_obs'] = int(np.sum([SENSOR_DIMS[sensor] for sensor in agent['obs_include']]))
    print("****************************************************************")
    print('Initial relative error:', np.sqrt(cost.evaluate(fcns[0]['init_loc']))/normb)

    gd_fcns = [{'fcn_obj': cost, 'dim': param_dim, 'init_loc': init_loc}]
    cg_fcns = [{'fcn_obj': cost, 'dim': param_dim, 'init_loc': init_loc}]
    lbfgs_fcns = [{'fcn_obj': cost, 'dim': param_dim, 'init_loc': init_loc}]
    mm_fcns = [{'fcn_obj': cost, 'dim': param_dim, 'init_loc': init_loc}]
    lr_fcns = [{'fcn_obj': cost, 'dim': param_dim, 'init_loc': init_loc}]
    #agent_gd = copy.deepcopy(agent)
    #agent_cg = copy.deepcopy(agent)
    #agent_lbfgs = copy.deepcopy(agent)
    #agent_mm = copy.deepcopy(agent)
    #agent_lr = copy.deepcopy(agent)
    for i in range(len(pols)):
        
        agent['fcns'] = gd_fcns
        Agent_gd = AgentLTO(agent)
        gd_pol = GradientDescentPolicy(Agent_gd, learning_rate, 0)
        agent['fcns'] = cg_fcns
        Agent_cg = AgentLTO(agent)
        cg_pol = ConjugateGradientPolicy(Agent_cg, learning_rate, 0)
        agent['fcns'] = lbfgs_fcns
        Agent_lbfgs = AgentLTO(agent)
        lbfgs_pol = LBFGSPolicy(Agent_lbfgs, learning_rate, mem_len, 0)
        agent['fcns'] = mm_fcns
        Agent_mm = AgentLTO(agent)
        mm_pol = MomentumPolicy(Agent_mm, learning_rate, momentum, 0)
        agent['fcns'] = lr_fcns
        Agent_lr = AgentLTO(agent)
        network_dir = exp_dir + 'data_files_pde/' + ('policy_itr_%02d' % pols[i]) + '.pkl'
        lr_pol = TfPolicy.load_policy(network_dir, first_derivative_network, network_config=network_config)

        x_gd = np.expand_dims(Agent_gd.sample(gd_pol, 0, verbose=False, save=False, noisy=False, usescale=False).get_X()[-1], axis=1)
        gd_fcns[0]['init_loc'] = x_gd
        print('Relative error after', agent['T'], 'iteration using GradientDescent Policy :', np.sqrt(cost.evaluate(x_gd))/normb)
        x_cg = np.expand_dims(Agent_cg.sample(cg_pol, 0, verbose=False, save=False, noisy=False, usescale=False).get_X()[-1], axis=1)
        cg_fcns[0]['init_loc'] = x_cg
        print('Relative error after', agent['T'], 'iteration using ConjuageGradient Policy:', np.sqrt(cost.evaluate(x_cg))/normb)
        x_lbfgs = np.expand_dims(Agent_lbfgs.sample(lbfgs_pol, 0, verbose=False, save=False, noisy=False, usescale=False).get_X()[-1], axis=1)
        lbfgs_fcns[0]['init_loc'] = x_lbfgs
        print('Relative error after', agent['T'], 'iteration using LBFGS Policy        :', np.sqrt(cost.evaluate(x_lbfgs))/normb)
        x_mm = np.expand_dims(Agent_mm.sample(mm_pol, 0, verbose=False, save=False, noisy=False, usescale=False).get_X()[-1], axis=1)
        mm_fcns[0]['init_loc'] = x_mm
        print('Relative error after', agent['T'], 'iteration using Momentum Policy        :', np.sqrt(cost.evaluate(x_mm))/normb)
        x_lr = np.expand_dims(Agent_lr.sample(lr_pol, 0, verbose=False, save=False, noisy=False, usescale=False).get_X()[-1], axis=1)
        lr_fcns[0]['init_loc'] = x_lr
        print('Relative error after', agent['T'], 'iteration using Learned Policy         :', np.sqrt(cost.evaluate(x_lr))/normb)

def main():
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str, help='experiment name')
    args = parser.parse_args()

    exp_name = args.experiment
    BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
    EXP_DIR = BASE_DIR + '/../experiments/' + exp_name + '/'
    hyperparams_file = EXP_DIR + 'hyperparams.py'
    gps_globals.phase = "TRAIN"
    hyperparams = imp.load_source('hyperparams', hyperparams_file)

    num = 6
    range_dim = range(4, 4 + num)
    #input_dim = [2**i for i in range_dim]
    input_dim = [int(2**3.5), int(2**4), int(2**4.5), int(2**5), int(2**5.5), int(2**6)]
    itr = hyperparams.config['algorithm']['iterations']
    pols = [0, 0, 1, 1]

    print('Comparation of solving a PDE using different policies starts: ')
    for i in range(num):
        print('Log2(problem size) = ', np.log(input_dim[i]**2)/np.log(2))
        pde_policy_comp(hyperparams, EXP_DIR, pols = pols, input_dim_1 = input_dim[i], input_dim_2 = input_dim[i], a = 0.1, b = 0.1,learning_rate = 0.0001, mem_len = 10, momentum = 0.9, err = 1e-4)
        print('===================================== Done =======================================')

if __name__ == '__main__':
    main()
