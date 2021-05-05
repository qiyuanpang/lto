#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import os
import os.path
from datetime import datetime
import numpy as np

import gps
from gps import __file__ as gps_filepath
from gps.agent.lto.agent_lto import AgentLTO
from gps.agent.lto.lto_world import LTOWorld
from gps.algorithm.algorithm import Algorithm
from gps.algorithm.cost.cost import Cost
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_lr_fixed import DynamicsLRFixed
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.traj_opt.traj_opt import TrajOpt
from gps.algorithm.traj_opt.traj_opt_modified import TrajOptMod
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.lto_model import fully_connected_tf_network, fully_connected_tf_network_leaky_relu, fully_connected_tf_network_swish
from gps.algorithm.policy.lin_gauss_init import init_lto_controller
from gps.proto.gps_pb2 import CUR_LOC, PAST_OBJ_VAL_DELTAS, PAST_GRADS, CUR_GRAD, PAST_LOC_DELTAS, ACTION
#from gps.agent.lto.fcn import LogisticRegressionFcnFamily, LogisticRegressionFcn
from gps.agent.lto.fcn import RobustRegressionFcnFamily, RobustRegressionFcn
from gps.agent.lto.fcn import QuadraticNormFcnFamily, QuadraticNormFcn, TraceNormFcnFamily, TraceNormFcn, QuadraticNormCompFcnFamily, QuadraticNormCompFcn
from gps.algorithm.cost.cost_utils import RAMP_CONSTANT
from gps.algorithm.policy_opt.lto_model import first_derivative_network, first_derivative_network_leaky_relu, first_derivative_network_swish
from gps.pdesolvers import f_fcn, g_fcn
from gps.elescatter import generate_Z, HSSBF_Zfun, findeq, findge
from gps.utility1 import CGM

#import sys
#from importlib import reload
#reload(sys)
#sys.setdefaultencoding('utf-8')

try:
   import cPickle as pickle
except:
   import pickle
import copy


#def f_fcn(x, y):
#    return x*y+1.0

#def g_fcn(x, y, k):
#    return np.log(np.square(1+x+1.0/k) + np.square(y))

def gen_fcns(interval1, interval2, max_dim_1, max_dim_2, num_fcns, session, gpu_id = 0, num_inits_per_fcn = 1, num_points_per_class = 50):

    max_dim = (max_dim_1-2)*(max_dim_2-2)
    max_dim1 = max_dim_1*max_dim_2
    fcn_family = QuadraticNormFcnFamily(max_dim, gpu_id = gpu_id, session = session)

    param_dim = fcn_family.get_total_num_dim()

    fcn_objs = []
    dims = np.random.randint(int(2*max_dim_1/3), max_dim_1, size=num_fcns)
    #print('dims = ', dims, max_dim_1)
    init_locs = np.random.randn(param_dim,num_fcns*num_inits_per_fcn)
    if 1:
       for k in range(num_fcns):
           
           dim = dims[k]
           dim1 = dim - 2
           dx = (interval1[1]-interval1[0])/dim
           dy = (interval2[1]-interval2[0])/dim
           dx2 = 1/(dx*dx)
           dy2 = 1/(dy*dy)
           mx = np.max([dx2, dy2, 2*(dx2+dy2)])
           x_pts = np.arange(interval1[0], interval1[1]+dx/2,dx)
           y_pts = np.arange(interval2[0], interval2[1]+dy/2,dy)
           
           ind1 = []
           ind2 = []
           data = np.zeros([dim*dim, dim*dim])
           labels = np.zeros([dim*dim,1])
           k_g = np.random.rand(1)[0]+0.5
           k_f = np.random.rand(1)[0]+0.5
           for j in range(0, dim):
               for i in range(0, dim):
                   if i == 0 or j == dim - 1 or i == dim - 1 or j == 0:
                      ind2.append(j*dim+i)
                      data[j*dim + i, j*dim + i] = 1.0
                      labels[j*dim + i] = g_fcn(x_pts[i], y_pts[j], k_g)
                   else:
                      ind1.append(j*dim+i)
                      data[j*dim + i, j*dim + i - 1] = dx2
                      data[j*dim + i, j*dim + i + 1] = dx2
                      data[j*dim + i, (j-1)*dim + i] = dy2
                      data[j*dim + i, (j+1)*dim + i] = dy2
                      data[j*dim + i, j*dim + i] = -2*(dx2 + dy2)
                      labels[j*dim + i] = f_fcn(x_pts[i], y_pts[j], k_f)
           #data = data/mx
           #labels = labels/mx

           data1 = data[ind1,:][:,ind1]
           data2 = data[ind1,:][:,ind2]
           labels1 = labels[ind1,:] - np.matmul(data2,labels[ind2,:])
        
           data1 = -data1/mx
           labels1 = -labels1/mx
           
           #print(type(data1), len(data1), len(data1[0]))
           data = np.eye(max_dim)*data1.max()
           labels = np.zeros([max_dim, 1])
           data[0:dim1*dim1, 0:dim1*dim1] = data1
           labels[0:dim1*dim1] = labels1
        
           fcn = QuadraticNormFcn(fcn_family, data, labels, disable_subsampling = True)
           for j in range(num_inits_per_fcn):
              init_locs[:, k*num_inits_per_fcn+j] = CGM(data, labels, np.vstack((np.random.randn(dim1*dim1, 1), np.zeros([max_dim-dim1*dim1, 1]))), 8)[:,0]
              #print(init_locs[:, k*num_inits_per_fcn+j])
              fcn_objs.append(fcn)
       
    #if eig < 0 and pde > 0:
    fcns = [{'fcn_obj': fcn_objs[k], 'dim': param_dim, 'init_loc': init_locs[:,k][:,None]} for k in range(num_fcns*num_inits_per_fcn)]
    #fcns = [{'fcn_obj': fcn_objs[k], 'dim': param_dim, 'init_loc': init_locs[:,k][:,None]} for k in range(num_fcns*num_inits_per_fcn)]

    return fcns,fcn_family


'''
def gen_fcns(input_dim, num_fcns, session, num_inits_per_fcn = 1, num_points_per_class = 50):

    #fcn_family = RobustRegresssionFcnFamily(input_dim, gpu_id = 0, session = session, tensor_prefix = "logistic_reg")
    fcn_family = RobustRegressionFcnFamily(input_dim, gpu_id = 0, session = session)
    # Dimensionality of the space over which optimization is performed
    param_dim = fcn_family.get_total_num_dim()

    fcn_objs = []

    for i in range(num_fcns):
        
        data = []
        for j in range(2):
            mu = np.random.randn(input_dim)
            sigma = np.random.randn(input_dim, input_dim)
            sigma_sq = np.dot(sigma, sigma.T)
            data.append(np.random.multivariate_normal(mu, sigma_sq, num_points_per_class))
        data = np.vstack(data)
        labels = np.vstack((np.zeros((num_points_per_class,1),dtype=np.int),np.ones((num_points_per_class,1),dtype=np.int)))
        
        fcn = RobustRegressionFcn(fcn_family, data, labels, 0.1, disable_subsampling = True)
        for j in range(num_inits_per_fcn):
            fcn_objs.append(fcn)

    init_locs = np.random.randn(param_dim,num_fcns*num_inits_per_fcn) 

    fcns = [{'fcn_obj': fcn_objs[i], 'dim': param_dim, 'init_loc': init_locs[:,i][:,None]} for i in range(num_fcns*num_inits_per_fcn)]
    
    return fcns,fcn_family
'''
def lto_on_exit(config):
    config['agent']['fcn_family'].destroy()

gpu_id = 0
gpu_ids = [0, 1, 2, 3, 4, 5, 6, 8]
#gpus = ''
#for i in range(len(gpu_ids)): gpus += str(gpu_ids[i])
#os.environ['CUDA_VISIBLE_DEVICES'] = gpus

a = 0.1
b = 0.1
max_dim_1 = 12
max_dim_2 = 12
#dx = a/input_dim_1
#dy = b/input_dim_2
#x_pts = np.arange(0,a+dx/2,dx)
#y_pts = np.arange(0,b+dy/2,dy)

#session = tf.Session()
session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True))
history_len = 10

num_fcns = 5
#input_dim = (input_dim_1-2)*(input_dim_2-2)

cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_file = cur_dir + "/trainset_pde.pkl"

if os.path.isfile(dataset_file):
    print("Dataset already exists. Loading from %s. " % (dataset_file))
    with open(dataset_file, mode="rb") as f:
        fcns,fcn_family = pickle.load(f)
    fcn_family.start_session(session)
else:
    print("Generating new dataset.")
    #fcns,fcn_family = gen_fcns(input_dim, num_fcns, session)
    function = lambda sess: gen_fcns([0, a], [0, b], max_dim_1, max_dim_2, num_fcns, sess, gpu_id, num_inits_per_fcn = 1)
    fcns,fcn_family = function(session)
    with open(dataset_file, mode="wb") as f:
        pickle.dump((fcns,fcn_family), f)
    print("Saved to %s. " % (dataset_file))
    
param_dim = fcns[0]['dim']

#print(param_dim, input_dim)
#assert param_dim == input_dim

SENSOR_DIMS = { 
    CUR_LOC: param_dim,
    PAST_OBJ_VAL_DELTAS: history_len,
    PAST_GRADS: history_len*param_dim,
    PAST_LOC_DELTAS: history_len*param_dim,
    CUR_GRAD: param_dim, 
    ACTION: param_dim
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/laplace/'

data_files_dir = EXP_DIR + 'data_files_pde/'
log_filename = EXP_DIR + 'log_pde.txt'

#print('!!!!!!!!!!!!!!!!!!!',fcns,fcn_family)

common = {
    'experiment_name': 'laplace' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': data_files_dir,
    'log_filename': log_filename,
    'conditions': num_fcns
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'gen_fcns': function,
    'type': AgentLTO,
    'world' : LTOWorld,
    'substeps': 1,
    'conditions': common['conditions'],
    'dt': 0.05,
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [CUR_LOC],
    'obs_include': [PAST_OBJ_VAL_DELTAS, PAST_GRADS, CUR_GRAD, PAST_LOC_DELTAS],
    'history_len': history_len,
    'fcns': fcns,
    'fcn_family': fcn_family     # Only used to destroy these at the end
}

algorithm = {
    'type': Algorithm,
    'conditions': common['conditions'],
    'iterations': 2,
    'inner_iterations': 2,
    'policy_dual_rate': 0.2, 
    'init_pol_wt': 0.01, 
    'ent_reg_schedule': 0.0, ## why set this to be 0 ?? this should not be zero, \mu_t actually.
    'fixed_lg_step': 3,
    'kl_step': 0.2, 
    'min_step_mult': 0.01, 
    'max_step_mult': 10.0, 
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
    'policy_sample_mode': 'replace',
    'num_of_streams': 8,
    'threads_per_block': 128
}

algorithm['init_traj_distr'] = {
    'type': init_lto_controller,
    'init_var': 0.01, 
    'dt': agent['dt'],
    'T': agent['T'],
    'all_possible_momentum_params': np.array([0.82, 0.84, 0.86, 0.88, 0.9, 0.92]),
    'all_possible_learning_rates': np.array([0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
}

algorithm['cost'] = {
    'type': Cost,
    'ramp_option': RAMP_CONSTANT, 
    'wp_final_multiplier': 1.0, 
    'weight': 1.0,
}

algorithm['dynamics'] = {
    'type': DynamicsLRFixed,
    'regularization': 1e-3,     # Increase this if Qtt is not PD during DGD
    'clipping_thresh': None,
    'prior': {
        'type': None, #DynamicsPriorGMM,
        'max_clusters': 20, 
        'min_samples_per_cluster': 20,
        'max_samples': 20,
        'strength': 1.0         # How much weight to give to prior relative to samples
    }
}

algorithm['traj_opt'] = {
    'type': TrajOptMod,
}

algorithm['policy_opt'] = {
    'type': PolicyOpt,
    'network_model': first_derivative_network,
    'iterations': 5, 
    'init_var': 0.01, 
    'batch_size': 64*len(gpu_ids),
    'solver_type': 'adam',
    'lr': 0.001, 
    'lr_policy': 'fixed',
    'momentum': 0.9,
    'weight_decay': 0.005,
    'use_gpu': 1,
    'model_dir': EXP_DIR + 'logs/',
    'gpu_ids': gpu_ids,
    'weights_file_prefix': EXP_DIR + 'policy',
    'policy_dict_path': None,
    'network_params': {
        'obs_include': agent['obs_include'],
        'sensor_dims': agent['sensor_dims'],
        'dim_hidden': [10, 10, 10],
        'history_len': history_len,
        'param_dim': param_dim,
        'weights_prev': None,
        'biases_prev': None,
        'gpu_ids': gpu_ids,
        'momentum': 0.9,
        'momentum2': 0.99,
        'lr': 0.001,
        'solver_type': 'adam',
        'epsilon': 1e-8
    }
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20, 
    'min_samples_per_cluster': 20, 
    'max_samples': 20,
    'strength': 1.0,
    'clipping_thresh': None, 
    'init_regularization': 1e-3, 
    'subsequent_regularization': 1e-3 
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 4,
    'common': common,
    'agent': agent,
    'algorithm': algorithm,
    'on_exit': lto_on_exit,
}
