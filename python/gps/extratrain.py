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
from gps.agent.lto.fcn import QuadraticNormFcnFamily, QuadraticNormFcn, TraceNormFcnFamily, TraceNormFcn, QuadraticNormCompFcnFamily, QuadraticNormCompFcn
from gps.algorithm.cost.cost_utils import RAMP_CONSTANT
from gps.gps_main import GPSMain
from gps.pdesolvers import f_fcn, g_fcn, act, def_cost, pdesolver
from gps.eigfinders import eigfinder


#def f_fcn(x, y):
#    return 0.0

#def g_fcn(x, y, k):
#    return np.log(np.square(1+x+1.0/k) + np.square(y))

def gen_fcns_more(dx, dy, x_pts, y_pts, input_dim_1, input_dim_2, num_fcns, session, exp_dir, itrs, pde_flag, eig_flag, obs_flag = None, num_inits_per_fcn = 1, num_points_per_class = 50):

    input_dim = (input_dim_1-2)*(input_dim_2-2)
    input_dim1 = input_dim_1*input_dim_2
    #print input_dim1
    #print eig, pde
    if eig_flag < 0 and pde_flag > 0:
       fcn_family = QuadraticNormFcnFamily(input_dim, gpu_id = 0, session = session)
    elif eig_flag > 0 and pde_flag < 0:
       fcn_family = TraceNormFcnFamily(input_dim, gpu_id = 0, session = session)

    param_dim = fcn_family.get_total_num_dim()

    fcn_objs = []

    dx2 = 1/(dx*dx)
    dy2 = 1/(dy*dy)
    mx = np.max([dx2, dy2, 2*(dx2+dy2)])
    for k in range(num_fcns):
        ind1 = []
        ind2 = []
        data = np.zeros([input_dim1, input_dim1])
        labels = np.zeros([input_dim1,1])
        k_g = np.random.rand(1)[0]+0.5
        k_f = np.random.rand(1)[0]+0.5
        for j in range(0, input_dim_2):
            for i in range(0, input_dim_1):
                if i == 0 or j == input_dim_2 - 1 or i == input_dim_1 - 1 or j == 0:
                    ind2.append(j*input_dim_1+i)
                    data[j*input_dim_1 + i, j*input_dim_1 + i] = 1.0
                    labels[j*input_dim_1 + i] = g_fcn(x_pts[i], y_pts[j], k_g)
                else:
                    ind1.append(j*input_dim_1+i)
                    data[j*input_dim_1 + i, j*input_dim_1 + i - 1] = dx2
                    data[j*input_dim_1 + i, j*input_dim_1 + i + 1] = dx2
                    data[j*input_dim_1 + i, (j-1)*input_dim_1 + i] = dy2
                    data[j*input_dim_1 + i, (j+1)*input_dim_1 + i] = dy2
                    data[j*input_dim_1 + i, j*input_dim_1 + i] = -2*(dx2 + dy2)
                    labels[j*input_dim_1 + i] = f_fcn(x_pts[i], y_pts[j], k_f)
        data1 = data[ind1,:][:,ind1]
        data2 = data[ind1,:][:,ind2]
        labels1 = labels[ind1,:] - np.matmul(data2, labels[ind2, :])
        #data1 = data1/mx
        #labels1 = labels1/mx
        
        if eig_flag < 0 and pde_flag > 0:
           data1 = data1/mx
           labels1 = labels1/mx
           fcn = QuadraticNormFcn(fcn_family, data1, labels1, disable_subsampling = True)
        elif eig_flag > 0 and pde_flag < 0:
           data1 = data1/mx/mx
           labels1 = labels1/mx/mx
           e,v = np.linalg.eig(data1)
           eig = np.min(e)
           fcn = TraceNormFcn(fcn_family, data1, labels1, eig, disable_subsampling = True)
        for j in range(num_inits_per_fcn):
           fcn_objs.append(fcn)
    init_locs = np.zeros([param_dim,num_fcns*num_inits_per_fcn])
    #print param_dim
    pol = None
    #b = np.zeros(param_dim)
    #print eig, pde
    for k in range(num_fcns*num_inits_per_fcn):
        if eig_flag < 0 and pde_flag > 0:
           b = np.squeeze(labels1)
           x = np.random.rand(param_dim)
           for j in range(len(obs_flag)):
               x = pdesolver(x, b, pol, fcn_objs[k], err = 1e-7, it_max = 50, obs_flag = obs_flag[j])
               x = x[0]
        elif eig_flag > 0 and pde_flag < 0:
           b = np.zeros(param_dim)
           #x = np.random.rand(param_dim)
           #x = x/np.linalg.norm(x)
           x = np.zeros(param_dim)
           for j in range(len(obs_flag)):
               x = eigfinder(x, b, pol, fcn_objs[k], err = 1e-7, it_max = 50, obs_flag = obs_flag[j])
               x = x[0]
        #x = pdesolver(x[0], pol, fcn_objs[k], err = 1e-7, it_max = 50, obs_flag = obs_flag
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis = 1)
        init_locs[:,k][:,None] = x
        #print 2-np.matmul(np.transpose(x), x)
        #print (2-np.matmul(np.transpose(x), x))*(np.matmul(np.transpose(x), np.matmul(data1, x)))-eig
        #print np.squeeze(2-np.matmul(np.transpose(x), x))*np.matmul(data1+np.transpose(data1), x)-2*x*np.squeeze(np.matmul(np.transpose(x), np.matmul(data1, x)))-eig
    #print(num_fcns*num_inits_per_fcn)
    #print(np.size(labels),labels)
    #print(np.size(data),data[18,:][:,None])
    if eig_flag < 0 and pde_flag > 0:
       fcns = [{'fcn_obj': fcn_objs[k], 'dim': param_dim, 'init_loc': init_locs[:,k][:,None]} for k in range(num_fcns*num_inits_per_fcn)]
    elif eig_flag > 0 and pde_flag < 0:
       fcns = [{'fcn_obj': fcn_objs[k], 'dim': param_dim, 'init_loc': init_locs[:,k][:,None], 'eig': eig} for k in range(num_fcns*num_inits_per_fcn)]

    return fcns,fcn_family


def train_more(config, exp_dir, input_dim_1, input_dim_2, itrs, session, pde, eig, a=1.0, b=1.0):

    #session = tf.Session()
    #cond = cond_init
    #print '2', eig, pde
    assert len(input_dim_2) == len(input_dim_1)
    for i in range(len(input_dim_1)):

    	dx = a/input_dim_1[i]
        dy = b/input_dim_2[i]
        x_pts = np.arange(0,a+dx/2,dx)
        y_pts = np.arange(0,b+dy/2,dy)

        param_dim = (input_dim_1[i]-2)*(input_dim_2[i]-2)
        history_len = config['agent']['history_len']

        SENSOR_DIMS = { 
            CUR_LOC: param_dim,
            PAST_OBJ_VAL_DELTAS: history_len,
            PAST_GRADS: history_len*param_dim,
            PAST_LOC_DELTAS: history_len*param_dim,
            CUR_GRAD: param_dim, 
            ACTION: param_dim
        }
        policy_dirs = []
        if eig < 0 and pde > 0:
           for c in range(len(itrs)):
               #if c < len(itrs)-1:
                  policy_dirs.append(exp_dir + 'data_files_pde' + '/policy_itr_' + '%02d' % itrs[c] + '_tf_data_' + '%02d' % c + '.ckpt')
               #else:
                  #policy_dirs.append(exp_dir + 'data_files_pde' + '/policy_itr_' + '%02d' % itrs[c] + '_tf_data.ckpt')
        elif eig > 0 and pde < 0:
           for c in range(len(itrs)):
               #if c < len(itrs)-1:
                  policy_dirs.append(exp_dir + 'data_files_eig' + '/policy_itr_' + '%02d' % itrs[c] + '_tf_data_' + '%02d' % c + '.ckpt')
               #else:
                  #policy_dirs.append(exp_dir + 'data_files_eig' + '/policy_itr_' + '%02d' % itrs[c] + '_tf_data.ckpt')
        #saver = tf.train.import_meta_graph(policy_dir + '.meta')
        #saver.restore(session, policy_dir)
        reader = tf.train.NewCheckpointReader(policy_dirs[-1])
        variables = reader.get_variable_to_shape_map()
        dirs_num = len(policy_dirs)
        if dirs_num == 1:
            ax = ''
        else:
            ax = '_' + '%1d' % 1
        config['algorithm']['policy_opt']['network_params']['weights_prev'] = [reader.get_tensor('w_0'+ax)] + [reader.get_tensor('w_1'+ax)]
        config['algorithm']['policy_opt']['network_params']['biases_prev'] = [reader.get_tensor('b_0'+ax)] + [reader.get_tensor('b_1'+ax)]

        config['agent']['sensor_dims'] = SENSOR_DIMS
        config['algorithm']['policy_opt']['network_params']['sensor_dims'] = SENSOR_DIMS
        config['algorithm']['policy_opt']['network_params']['param_dim'] = param_dim
        num_fcns = config['common']['conditions']
        
        obs_flag = []
        for j in range(dirs_num):
            if j == 0:
                ax = ''
            else:
                ax = '_' + '%1d' % 1
            reader = tf.train.NewCheckpointReader(policy_dirs[j])
            weights = [reader.get_tensor('w_0'+ax)] + [reader.get_tensor('w_1'+ax)]
            biases = [reader.get_tensor('b_0'+ax)] + [reader.get_tensor('b_1'+ax)]
            obs_flag.append({'sensor_dims': SENSOR_DIMS, 'param_dim': param_dim, 
                  'obs_include': config['algorithm']['policy_opt']['network_params']['obs_include'],
                  'history_len': history_len,
                  'net_weights': weights,
                  'net_biases': biases
            })
        #with tf.Session() as sess:
        #print '3', eig, pde
        fcns, fcn_family = gen_fcns_more(dx, dy, x_pts, y_pts, input_dim_1[i], input_dim_2[i], num_fcns, session, exp_dir, itrs, pde, eig, obs_flag)
        config['agent']['fcns'] = fcns
        config['agent']['fcn_family'] = fcn_family

        gps = GPSMain(config)
        #print '***************************** Extra train ' + '%02d' % i + ', Starts ********************************************'
        gps.run()
        #cond = np.where(cs==np.min(cs))[0][0]
        #print '***************************** Extra train ' + '%02d' % i + ', Ends!  ********************************************'
        


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

    #seed = hyperparams.config.get('random_seed', 0)
    #random.seed(seed)
    #np.random.seed(seed)
   
    #hyperparams.config['algorithm']['init_traj_distr']['all_possible_momentum_params'] = np.array([0.16, 0.18, 0.20, 0.22, 0.24, 0.26])
    #hyperparams.config['algorithm']['init_traj_distr']['all_possible_momentum_params'] = np.array([0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064])
    #hyperparams.config['algorithm']['policy_opt']['lr'] = 0.0005
    #hyperparams.config['algorithm']['policy_opt']['momentum'] = 0.3
    #hyperparams.config['algorithm']['weight_decay'] = 0.003
    #hyperparams.config['algorithm']['use_gpu'] = 0

    hyperparams.config['algorithm']['init_traj_distr']['all_possible_momentum_params'] = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.14])
    hyperparams.config['algorithm']['init_traj_distr']['all_possible_momentum_params'] = np.array([0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032])
    hyperparams.config['algorithm']['policy_opt']['lr'] = 0.0005
    hyperparams.config['algorithm']['policy_opt']['momentum'] = 0.3
    hyperparams.config['algorithm']['weight_decay'] = 0.003
    hyperparams.config['algorithm']['use_gpu'] = 0
    hyperparams.config['algorithm']['iterations'] = 100

    #session = hyperparams.session
    config = hyperparams.config
    history_len = config['agent']['history_len']
    itr = config['algorithm']['iterations']
    #session = tf.Session()
 
    dim_min = 7
    dim_max = 16
    step = 2
    dims = range(dim_min, dim_max, step)
    eig = -1.0
    pde = 1.0
    #K = np.random.randint(1, len(dims)*2, len(dims))*1.0
    itrs = [2, 3, 1] #pde
    #itrs = [1, 7] #eig
    print 'Extra training Begins:'
    for i in range(len(dims)):
        session = tf.Session()
        config = copy.deepcopy(hyperparams.config)
        #config = hyperparams.config
        print '***************************** Extra train ' + '%02d' % i + ', Starts ********************************************'
        train_more(config, exp_dir, [dims[i]], [dims[i]], itrs, session, pde, eig)
        print '***************************** Extra train ' + '%02d' % i + ', Ends!  ********************************************'
    print 'Extra training Ends!'

if __name__ == "__main__":
    main()

