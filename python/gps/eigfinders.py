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
from gps.agent.lto.fcn import TraceNorm2FcnFamily, TraceNorm2Fcn, LagrangianFcnFamily4Eig, LagrangianFcn4Eig, LagrangianFcnFamily4Eig1, LagrangianFcn4Eig1
from gps.algorithm.cost.cost_utils import RAMP_CONSTANT


def f_fcn(x, y, k):
    return x*(y+1.0/k)

def g_fcn(x, y, k):
    return np.log(np.square(1+x+1.0/k) + np.square(y))

def relu(mat):
    m = copy.deepcopy(mat)
    n = len(m)
    #print(m)
    for i in range(n):
        m[i] = np.max(mat[i], 0)
    return m

def principaleig(A, v):
    w = np.matmul(A, v)
    y = v.dot(w)
    x = v.dot(v)
    eig = y/x
    err = np.linalg.norm(w-eig*v)/np.linalg.norm(w)
    return eig, err

def act(obs, config):
    sensor_dims = config['sensor_dims']
    obs_include = config['obs_include']
    history_len = config['history_len']
    param_dim = config['param_dim']
    #print(param_dim,'??')
    weights = config['net_weights']
    biases = config['net_biases']
    obs_idx, dO = [], 0
    for sensor in obs_include:
        dim = sensor_dims[sensor]
        obs_idx.append(list(range(dO, dO+dim)))
        dO += dim
    assert len(obs) == dO

    n_layers = len(weights)
    cur_top = obs
    for layer_step in range(n_layers):
        top = {}
        shape = np.shape(weights[layer_step])
        #print(type(shape), shape)
        for i in range(shape[1]):
            top['slice_'+str(i)] = weights[layer_step][0, i]*cur_top[0:param_dim]
            for j in range(1, shape[0]):
                loc = cur_top[param_dim*j: param_dim*(j+1)]
                top['slice_'+str(i)] = np.add(top['slice_'+str(i)], weights[layer_step][j,i]*loc)
            top['slice_'+str(i)] = top['slice_'+str(i)] + biases[layer_step][i]
        cur_top = top['slice_'+str(0)]
        for i in range(1, shape[1]):
            #print(cur_top, top['slice_'+str(i)])
            cur_top = np.hstack([cur_top, top['slice_'+str(i)]])
        #print(np.shape(cur_top))
        if layer_step != n_layers - 1:
            cur_top = relu(cur_top)
    #print(len(cur_top))
    return cur_top


def def_cost(dx, dy, x_pts, y_pts, input_dim_1, input_dim_2, k, session):

    input_dim = (input_dim_1-2)*(input_dim_2-2)
    input_dim1 = input_dim_1*input_dim_2
    #print input_dim, input_dim1
    fcn_family = LagrangianFcnFamily4Eig(input_dim+1, gpu_id = 0, session = session)

    param_dim = fcn_family.get_total_num_dim()
    #print param_dim
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
    labels1 = labels[ind1,:] - np.matmul(data2,labels[ind2,:])
    data1 = -data1/mx
    labels1 = -labels1/mx
    eig = -2.0
    #print data1.shape, labels1.shape
    fcn = LagrangianFcn4Eig(fcn_family, data1, labels1, disable_subsampling = True)
    #for j in range(num_inits_per_fcn):
    #    fcn_objs.append(fcn)

    return fcn, fcn_family, labels1, data1

def eigfinder(x_init, b, pol, cost, err = 1e-4, it_max = 1000, obs_flag = None):
    t = 0
    nb = 1.0
    #nb = np.linalg.norm(b)**2
    x = copy.deepcopy(x_init)
    x_ep = np.expand_dims(x,axis=1)
    #print x_ep.shape
    #print x_ep, cost.evaluate(x)
    acc = [cost.evaluate(x_ep)/nb]
    if obs_flag == None:
        obs = None
        state = x
    else:
        sensor_dims = obs_flag['sensor_dims']
        obs_include = obs_flag['obs_include']
        obs_idx, dO = [], 0
        for sensor in obs_include:
            dim = sensor_dims[sensor]
            obs_idx.append(list(range(dO, dO+dim)))
            dO += dim
        obs = np.zeros(dO)
        param_dim = obs_flag['param_dim']
        history_len = obs_flag['history_len']
        dim = 0
        for sensor in obs_include:
            if sensor == CUR_GRAD:
                #print(sensor, sensor_dims[sensor], x_ep.shape)
                obs[dim:dim+sensor_dims[sensor]] = np.squeeze(cost.grad(x_ep), axis=1)
            else:
                obs[dim:dim+sensor_dims[sensor]] = 0.0
            dim += sensor_dims[sensor]
        state = np.hstack([x, obs])
        LOCS = deque(maxlen=history_len)
        LOCS.append(x)
        GRADS = deque(maxlen=history_len)
        GRADS.append(np.squeeze(cost.grad(x_ep), axis=1))
    
    while cost.evaluate(x_ep)/nb > err and t<it_max:
        #print(obs.shape, obs)
        if obs_flag == None:
           dx = pol.act(state, obs, t, noise = None)
        else:
           #print(len(obs))
           dx = act(obs, obs_flag)
        #if pol == None:
        #print(obs)
        #   print(t, np.shape(dx), np.shape(x))
        x = x + dx
        x_ep = np.expand_dims(x,axis=1)
        acc.append((cost.evaluate(x_ep)/nb))
        t += 1
        if obs_flag is None:
            state = x
        else:
            obs = np.zeros(dO)
            #print(LOCS)
            locs = copy.deepcopy(LOCS)
            grads = copy.deepcopy(GRADS)
            #objs = copy.deepcopy(OBJS)
            qsize = len(locs)
            for i in range(qsize):
                #obs[qsize-1-i] = objs.popleft() - acc[-1]
                obs[(qsize-1-i)*param_dim:(qsize-i)*param_dim] = grads.popleft()
                obs[history_len*param_dim:(history_len+1)*param_dim] = np.squeeze(cost.grad(x_ep), axis=1)
                obs[(history_len+1)*param_dim+(qsize-1-i)*param_dim:(history_len+1)*param_dim+(qsize-i)*param_dim] = locs.popleft() - x
            if len(LOCS) == LOCS.maxlen:
                LOCS.popleft()
                GRADS.popleft()
                #OBJS.popleft()
            else:
                #obs[qsize:history_len] = 0.0
                obs[qsize*param_dim:history_len*param_dim] = 0.0
                obs[(history_len+1+qsize)*param_dim:dO] = 0.0
            LOCS.append(x)
            GRADS.append(np.squeeze(cost.grad(x_ep), axis=1))
            #OBJS.append(acc[-1])
            state = np.hstack([x,obs])
        
    return x, acc, t

def eig_policy_comp(hyperparams, exp_dir, pols, input_dim_1 = 128, input_dim_2 = 128, a = 1.0, b = 1.0, learning_rate = 0.0001, mem_len = 10, momentum = 0.9, err = 1e-4, it_max = 50):

    dx = a/input_dim_1
    dy = b/input_dim_2
    x_pts = np.arange(0,a+dx/2,dx)
    y_pts = np.arange(0,b+dy/2,dy)
    k = np.random.rand(1)[0]+0.5
    session = tf.Session()

    cost, fcn_family, b, data = def_cost(dx, dy, x_pts, y_pts, input_dim_1, input_dim_2, k, session)
    
    obs_flag = hyperparams.config['algorithm']['policy_opt']['network_params']

    history_len = hyperparams.config['agent']['history_len']
    param_dim = (input_dim_1-2)*(input_dim_2-2)+1
    #print(param_dim)
    fcns = [{'fcn_obj': cost, 'dim': param_dim, 'init_loc': np.zeros([param_dim, 1])}]

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
            'state_include': [CUR_LOC],
            'obs_include': hyperparams.config['agent']['obs_include'],
            'history_len': history_len,
            'fcns': fcns,
            'fcn_family': fcn_family
    }

    
    num_pols = len(pols)
    policy_dirs = []
    for i in range(num_pols):
        #if i < num_pols-1:
        #   policy_dirs.append(exp_dir + 'data_files_eig' + '/policy_itr_' + '%02d' % pols[i] + '_tf_data_' + '%02d' % i + '.ckpt')
        #else:
           policy_dirs.append(exp_dir + 'data_files' + '/policy_itr_' + '%02d' % pols[i] + '_tf_data.ckpt')
    #policy_dir = exp_dir + 'data_files_eig' + '/policy_itr_' + '%02d' % cond + '_tf_data.ckpt'
    #policy_dir0 = exp_dir + 'data_files_eig' + '/policy_itr_' + '%02d' % 1 + '_tf_data_00.ckpt'
    #sess = tf.Session()
    #saver = tf.train.import_meta_graph(policy_dir + '.meta')
    #saver.restore(sess, policy_dir)

    #reader = tf.train.NewCheckpointReader(policy_dir)
    #variables = reader.get_variable_to_shape_map()
    #obs_flag['net_weights'] = [reader.get_tensor('w_0_1')] + [reader.get_tensor('w_1_1')]
    #obs_flag['net_biases'] = [reader.get_tensor('b_0_1')] + [reader.get_tensor('b_1_1')]
    #print param_dim, obs_flag['net_weights'][0].shape
    #obs_flag['net_weights'] = [(np.random.rand(11,20)*2-1)/1000, (np.random.rand(20,1)*2-1)/1000]
    #obs_flag['net_biases'] = [(np.random.rand(20)*2-1)/1000, (np.random.rand(1)*2-1)/1000]
   
    reader = tf.train.NewCheckpointReader(policy_dirs[0])
    count = 0
    variables = reader.get_variable_to_shape_map()
    w = 'w_' + '%1d' % count
    while w in variables:
        count += 1
        w = 'w_' + '%1d' % count

    obs_flag = []
    for j in range(num_pols):
        if j == 0:
           ax = ''
        elif j == 1:
           ax = '_' + '%1d' % 1
        else:
           ax = ''
        #print j
        reader = tf.train.NewCheckpointReader(policy_dirs[j])
        weights = []
        biases = []
        for k in range(count):
           weights = weights + [reader.get_tensor('w_' + '%1d' % k + ax)]
           biases = biases + [reader.get_tensor('b_' + '%1d' % k + ax)]
        #weights = [reader.get_tensor('w_0'+ax)] + [reader.get_tensor('w_1'+ax)]
        #biases = [reader.get_tensor('b_0'+ax)] + [reader.get_tensor('b_1'+ax)]
        obs_flag.append({'sensor_dims': SENSOR_DIMS, 'param_dim': param_dim,
                'obs_include': hyperparams.config['algorithm']['policy_opt']['network_params']['obs_include'],
                'history_len': history_len,
                'net_weights': weights,
                'net_biases': biases
            })
    #obs_flag0 = copy.deepcopy(obs_flag)
    #reader = tf.train.NewCheckpointReader(policy_dir0)
    #variables = reader.get_variable_to_shape_map()
    #obs_flag0['net_weights'] = [reader.get_tensor('w_0')] + [reader.get_tensor('w_1')]
    #obs_flag0['net_biases'] = [reader.get_tensor('b_0')] + [reader.get_tensor('b_1')]

    agent_gd = AgentLTO(agent)
    gd_pol = GradientDescentPolicy(agent_gd, 0.05, 0)
    agent_cg = AgentLTO(agent)
    cg_pol = ConjugateGradientPolicy(agent_cg, 0.0001, 0)
    #agent_lbfgs = AgentLTO(agent)
    #lbfgs_pol = LBFGSPolicy(agent_lbfgs, learning_rate, mem_len, 0)
    agent_mm = AgentLTO(agent)
    mm_pol = MomentumPolicy(agent_mm, 0.05, momentum, 0)
    lr_pol = None
    #print(param_dim)
    #x_init = np.random.rand(param_dim)
    #x_init = x_init/np.linalg.norm(x_init)
    #x_init = np.ones(param_dim)
    np.random.seed(0)
    x_init = np.random.normal(0, 0.01, size=param_dim)

    e,v = np.linalg.eig(data)
    print 'Real largest(norm) eigenvalue: ', np.max(e)

    x_gd, acc_gd, t_gd = eigfinder(x_init, b, gd_pol, cost, err, it_max)
    eig_gd, err_gd = principaleig(data, x_gd[:param_dim-1])
    print 'Eigs have been found using GradientDescentPolicy,   initial cost: ', '%.3e' % acc_gd[0], 'final cost: ', '%.3e' % acc_gd[-1], ' iteration_num: ', t_gd, ' eig1: ', x_gd[-1], ' eig: ', eig_gd, ' err: ', err_gd
    x_cg, acc_cg, t_cg = eigfinder(x_init, b, cg_pol, cost, err, it_max)
    eig_cg, err_cg = principaleig(data, x_cg[:param_dim-1])
    print 'Eigs have been found using ConjugateGradientPolicy, initial cost: ', '%.3e' % acc_cg[0], 'final cost: ', '%.3e' % acc_cg[-1], ' iteration_num: ', t_cg, ' eig1: ', x_cg[-1], ' eig: ', eig_cg, ' err: ', err_cg
    #x_lbfgs, acc_lbfgs, t_lbfgs = eigfinder(x_init, lbfgs_pol, cost, err, it_max)
    #print 'Your PDE has been solved using LBFGSPolicy,            initial accuray: ', '%.3e' % acc_lbfgs[0], 'final accuracy: ', '%.3e' % acc_lbfgs[-1], ' iteration_num: ', t_lbfgs
    x_mm, acc_mm, t_mm = eigfinder(x_init, b, mm_pol, cost, err, it_max)
    eig_mm, err_mm = principaleig(data, x_mm[:param_dim-1])
    print 'Eigs have been found using MomentumPolicy,          initial cost: ', '%.3e' % acc_mm[0], 'final cost: ', '%.3e' % acc_mm[-1], ' iteration_num: ', t_mm, ' eig1: ', x_mm[-1], ' eig: ', eig_mm, ' err: ', err_mm
    x_lr = x_init
    #print(param_dim)
    for j in range(num_pols):
        x_lr, acc_lr, t_lr = eigfinder(x_lr, b, lr_pol, cost, err, it_max, obs_flag[j])
        eig_lr, err_lr = principaleig(data, x_lr[:param_dim-1])
        print 'Eigs have been found using Learned Policy ' + '%1d' % j + ',        initial cost: ', '%.3e' % acc_lr[0], 'final cost: ', '%.3e' % acc_lr[-1], ' iteration_num: ', t_lr, ' eig1: ', x_lr[-1], ' eig: ', eig_lr, ' err: ', err_lr
    #x_lr = x_lr/np.linalg.norm(x_lr)
    #x_lr, acc_lr, t_lr = eigfinder(x_lr, b, lr_pol, cost, err, it_max, obs_flag)
    #eig_lr, err_lr = principaleig(data, x_lr)
    #print 'Eigs have been found using Learned Policy 1,        initial cost: ', '%.3e' % acc_lr[0], 'final cost: ', '%.3e' % acc_lr[-1], ' iteration_num: ', t_lr, ' eig: ', eig_lr, ' err: ', err_lr

    GD = {'eigvec': x_gd, 'eig': eig_gd, 'accuracy': acc_gd}
    CG = {'eigvec': x_cg, 'eig': eig_cg, 'accuracy': acc_cg}
    MM = {'eigvec': x_mm, 'eig': eig_mm, 'accuracy': acc_mm}
    LR = {'eigvec': x_lr, 'eig': eig_lr, 'accuracy': acc_lr}

    results = {'GD': GD, 'CG': CG, 'MM': MM, 'LR': LR}

    return results

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
    
    num = 7
    #range_dim = range(4, 4 + num)
    range_dim = [1.8, 2.2, 2.6, 3, 3.4, 3.8, 4.2]
    input_dim = [int(2**i) for i in range_dim]
    itr = hyperparams.config['algorithm']['iterations']
    pols = [0]

    print 'Comparation of finding eigenvalues using different policies starts: '
    results = {}
    for i in range(num):
        print 'Log2(problem size) = ', np.log(input_dim[i]**2)/np.log(2)
        results['log2_size_'+str(range_dim[i])] = eig_policy_comp(hyperparams, exp_dir, pols, input_dim_1 = input_dim[i], input_dim_2 = input_dim[i], a = 0.1, b = 0.1, learning_rate = 0.0001, mem_len = 10, momentum = 0.9, err = 1e-15, it_max = 150)
        print '===================================== Done ======================================='
    #with open('./results.json', 'w') as f:
    #    json.dump(results, f)
    np.save('results.npy', results)
    print 'results saved as ./results.npy !'

if __name__ == "__main__":
    main()
