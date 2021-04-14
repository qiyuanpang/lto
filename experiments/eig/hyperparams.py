#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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
from gps.agent.lto.fcn import QuadraticNormFcnFamily, QuadraticNormFcn, TraceNormFcnFamily, TraceNormFcn, QuadraticNormCompFcnFamily, QuadraticNormCompFcn, LagrangianFcnFamily4Eig, LagrangianFcn4Eig
from gps.algorithm.cost.cost_utils import RAMP_CONSTANT
from gps.algorithm.policy_opt.lto_model import first_derivative_network, first_derivative_network_leaky_relu, first_derivative_network_swish
from gps.pdesolvers import f_fcn, g_fcn
from gps.elescatter import generate_Z, HSSBF_Zfun, findeq, findge

try:
   import cPickle as pickle
except:
   import pickle
import copy


#def f_fcn(x, y):
#    return x*y+1.0

#def g_fcn(x, y, k):
#    return np.log(np.square(1+x+1.0/k) + np.square(y))

def gen_fcns(dx, dy, x_pts, y_pts, input_dim_1, input_dim_2, num_fcns, session, gpu_id = 0, num_inits_per_fcn = 1, num_points_per_class = 50):

    input_dim = (input_dim_1-2)*(input_dim_2-2)
    input_dim1 = input_dim_1*input_dim_2
    #if eig < 0 and pde > 0 and prec < 0:
    fcn_family = LagrangianFcnFamily4Eig(input_dim+1, gpu_id = gpu_id, session = session)
    #elif eig < 0 and pde > 0 and prec > 0:
    #   fcn_family = QuadraticNormCompFcnFamily(input_dim, gpu_id = gpu_id, session = session)
    #elif eig > 0 and pde < 0:
    #   fcn_family = TraceNormFcnFamily(input_dim, gpu_id = gpu_id, session = session)

    param_dim = fcn_family.get_total_num_dim()
    #print(param_dim)
    fcn_objs = []
    
    if 1:
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
           #data = data/mx
           #labels = labels/mx

           data1 = data[ind1,:][:,ind1]
           data2 = data[ind1,:][:,ind2]
           labels1 = labels[ind1,:] - np.matmul(data2,labels[ind2,:])
        
           data1 = -data1/mx
           labels1 = -labels1/mx
           #e, v = np.linalg.eig(data1)
           #e = e/2
           #data1 = np.matmul(v, np.matmul(np.diag(e), v))
           #print(mx) 
           #if eig < 0 and pde > 0:
              #data1 = data1/mx
              #labels1 = labels1/mx
           fcn = LagrangianFcn4Eig(fcn_family, data1, labels1, disable_subsampling = True)
           #elif eig > 0 and pde < 0:
              #data1 = data1/mx/mx
              #labels1 = labels1/mx/mx
           #   e,v = np.linalg.eig(data1)
           #   eig = np.min(data1)
           #   fcn = TraceNormFcn(fcn_family, data1, labels1, eig, disable_subsampling = True)
           for j in range(num_inits_per_fcn):
               fcn_objs.append(fcn)

       
    #init_locs = np.random.randn(param_dim-1,num_fcns*num_inits_per_fcn)
    #zeros = np.zeros([1, num_fcns*num_inits_per_fcn])
    #init_locs = np.vstack((init_locs, zeros))
    #norms = np.linalg.norm(init_locs, axis=0)
    #init_locs = init_locs/norms
    init_locs = np.zeros([param_dim, num_fcns*num_inits_per_fcn])
    p = np.expand_dims(init_locs[:param_dim-1, 0], axis=0)
    #print(p.shape)
    ppT = np.matmul(np.transpose(p), p)
    #print(ppT)
    #print(np.linalg.norm(data1 - np.transpose(data1)))
    print(np.trace(data1) - np.trace(np.matmul(data1, ppT)))
    #if eig < 0 and pde > 0 and prec > 0:
    #   init_locs = np.random.randn(param_dim,num_fcns*num_inits_per_fcn) + 1j*np.random.randn(param_dim,num_fcns*num_inits_per_fcn)
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

a = 0.1
b = 0.1
input_dim_1 = 8
input_dim_2 = 8
dx = a/input_dim_1
dy = b/input_dim_2
x_pts = np.arange(0,a+dx/2,dx)
y_pts = np.arange(0,b+dy/2,dy)

session = tf.Session()
history_len = 10


num_fcns = 10 #100

input_dim = (input_dim_1-2)*(input_dim_2-2)

cur_dir = os.path.dirname(os.path.abspath(__file__))
#if eig < 0 and pde > 0 and prec < 0:
dataset_file = cur_dir + "/trainset_eig.pkl"
#elif eig < 0 and pde > 0 and prec > 0:
#   dataset_file = cur_dir + "/trainset_pde_" + "%02d" % prec + ".pkl"
#elif eig > 0 and pde < 0:
#   dataset_file = cur_dir + "/trainset_eig.pkl"

if os.path.isfile(dataset_file):
    print("Dataset already exists. Loading from %s. " % (dataset_file))
    with open(dataset_file, "rb") as f:
        fcns,fcn_family = pickle.load(f)
    fcn_family.start_session(session)
else:
    print("Generating new dataset.")
    #fcns,fcn_family = gen_fcns(input_dim, num_fcns, session)
    fcns,fcn_family = gen_fcns(dx, dy, x_pts, y_pts, input_dim_1, input_dim_2, num_fcns, session, gpu_id)
    with open(dataset_file, "wb") as f:
        pickle.dump((fcns,fcn_family), f)
    print("Saved to %s. " % (dataset_file))
    
param_dim = fcns[0]['dim']
#print(param_dim)
assert param_dim == input_dim+1

SENSOR_DIMS = { 
    CUR_LOC: param_dim,
    PAST_OBJ_VAL_DELTAS: history_len,
    PAST_GRADS: history_len*param_dim,
    PAST_LOC_DELTAS: history_len*param_dim,
    CUR_GRAD: param_dim, 
    ACTION: param_dim
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/eig/'

#if eig < 0 and pde > 0 and prec < 0:
data_files_dir = EXP_DIR + 'data_files/'
log_filename = EXP_DIR + 'log_eig.txt'
#elif eig < 0 and pde > 0 and prec > 0:
#    data_files_dir = EXP_DIR + 'data_files_pde_' + '%02d' % prec + '/'
#    log_filename = EXP_DIR + 'log_pde_' + '%02d' % prec + '.txt'
#elif eig > 0 and pde < 0:
#    data_files_dir = EXP_DIR + 'data_files_eig/'
#    log_filename = EXP_DIR + 'log_eig.txt'

#print('!!!!!!!!!!!!!!!!!!!',fcns,fcn_family)

common = {
    'experiment_name': 'eig' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': data_files_dir,
    'log_filename': log_filename,
    'conditions': num_fcns
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentLTO,
    'world' : LTOWorld,
    'substeps': 1,
    'conditions': common['conditions'],
    'dt': 0.01,
    'T': 150,
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
    'iterations': 50,
    'inner_iterations': 4,
    'policy_dual_rate': 0.1, 
    'init_pol_wt': 0.01, 
    'ent_reg_schedule': 0.0,
    'fixed_lg_step': 3,
    'kl_step': 0.2, 
    'min_step_mult': 0.01, 
    'max_step_mult': 8.0, 
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
    'policy_sample_mode': 'replace'
}

algorithm['init_traj_distr'] = {
    'type': init_lto_controller,
    'init_var': 0.01, 
    'dt': agent['dt'],
    'T': agent['T'],
    'all_possible_momentum_params': np.array([0.82, 0.84, 0.86, 0.88, 0.9, 0.92]),
    'all_possible_learning_rates': np.array([0.04, 0.08, 0.16, 0.30, 0.60, 1.2, 1.6])
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
        'type': None,#DynamicsPriorGMM,
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
    'network_model': first_derivative_network_leaky_relu,
    'iterations': 100, 
    'init_var': 0.01, 
    'batch_size': 25,
    'solver_type': 'adam',
    'lr': 0.0001, 
    'lr_policy': 'fixed',
    'momentum': 0.7,
    'weight_decay': 0.005,
    'use_gpu': 1,
    'gpu_id': gpu_id,
    'weights_file_prefix': EXP_DIR + 'policy',
    'network_params': {
        'obs_include': agent['obs_include'],
        'sensor_dims': agent['sensor_dims'],
        'dim_hidden': [20, 20, 5],
        'history_len': history_len,
        'param_dim': param_dim,
        'weights_prev': None,
        'biases_prev': None,
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
    'num_samples': 20,
    'common': common,
    'agent': agent,
    'algorithm': algorithm,
    'on_exit': lto_on_exit,
}
