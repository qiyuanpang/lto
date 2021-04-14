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
from gps.agent.lto.fcn import QuadraticNormFcnFamily, QuadraticNormFcn, TraceNormFcnFamily, TraceNormFcn, QuadraticNormCompFcnFamily, QuadraticNormCompFcn
from gps.algorithm.cost.cost_utils import RAMP_CONSTANT
from gps.algorithm.policy_opt.lto_model import first_derivative_network, first_derivative_network_leaky_relu, first_derivative_network_swish
from gps.pdesolvers import f_fcn, g_fcn
from gps.elescatter import generate_Z, HSSBF_Zfun, findeq, findge

try:
   import cPickle as pickle
except:
   import pickle
import copy


def gen_fcns(input_dim, vec, num_fcns, session, gpu_id = 0, num_inits_per_fcn = 1, num_points_per_class = 50):

    assert input_dim == (len(vec)-1)//2 + 1
    fcn_family = QuadraticNormFcnFamily(input_dim, gpu_id = gpu_id, session = session)

    param_dim = fcn_family.get_total_num_dim()

    fcn_objs = []
    
    for k in range(num_fcns):
        
        data = np.zeros([input_dim, input_dim])
        for j in range(0, input_dim):
            for i in range(0, input_dim):
                data[j, i] = vec[i-j]
        mx = np.max(np.abs(vec))
        data = data/mx/mx
        labels = np.random.rand(input_dim)
        labels = labels/np.linalg.norm(labels)
        
        fcn = QuadraticNormFcn(fcn_family, data, labels, disable_subsampling = True)
        for j in range(num_inits_per_fcn):
            fcn_objs.append(fcn)
    
       
    init_locs = np.random.randn(param_dim, num_fcns*num_inits_per_fcn)
    fcns = [{'fcn_obj': fcn_objs[k], 'dim': param_dim, 'init_loc': init_locs[:,k][:,None]} for k in range(num_fcns*num_inits_per_fcn)]
    #fcns = [{'fcn_obj': fcn_objs[k], 'dim': param_dim, 'init_loc': init_locs[:,k][:,None]} for k in range(num_fcns*num_inits_per_fcn)]

    return fcns,fcn_family


def lto_on_exit(config):
    config['agent']['fcn_family'].destroy()


gpu_id = 1

n = 25
vec = [i*1.0 for i in range(n, -n, -1)]
vec.remove(0.0)

session = tf.Session()
history_len = 10


num_fcns = 20 #100

input_dim = n

cur_dir = os.path.dirname(os.path.abspath(__file__))
#if eig < 0 and pde > 0 and prec < 0:
dataset_file = cur_dir + "/trainset_tpz.pkl"

if os.path.isfile(dataset_file):
    print("Dataset already exists. Loading from %s. " % (dataset_file))
    with open(dataset_file, "rb") as f:
        fcns,fcn_family = pickle.load(f)
    fcn_family.start_session(session)
else:
    print("Generating new dataset.")
    #fcns,fcn_family = gen_fcns(input_dim, num_fcns, session)
    fcns,fcn_family = gen_fcns(n, vec, num_fcns, session, gpu_id)
    with open(dataset_file, "wb") as f:
        pickle.dump((fcns,fcn_family), f)
    print("Saved to %s. " % (dataset_file))
    
param_dim = fcns[0]['dim']

assert param_dim == input_dim

SENSOR_DIMS = { 
    CUR_LOC: param_dim,
    PAST_OBJ_VAL_DELTAS: history_len,
    PAST_GRADS: history_len*param_dim,
    PAST_LOC_DELTAS: history_len*param_dim,
    CUR_GRAD: param_dim, 
    ACTION: param_dim
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/lto2/'

#if eig < 0 and pde > 0 and prec < 0:
data_files_dir = EXP_DIR + 'data_files/'
log_filename = EXP_DIR + 'log.txt'

common = {
    'experiment_name': 'lto2' + '_' + \
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
    'dt': 0.05,
    'T': 50,
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
    'iterations': 100,
    'inner_iterations': 10,
    'policy_dual_rate': 0.2, 
    'init_pol_wt': 0.01, 
    'ent_reg_schedule': 0.0,
    'fixed_lg_step': 3,
    'kl_step': 0.2, 
    'min_step_mult': 0.01, 
    'max_step_mult': 10.0, 
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
    'network_model': fully_connected_tf_network,
    'iterations': 200, 
    'init_var': 0.01, 
    'batch_size': 25,
    'solver_type': 'adam',
    'lr': 0.001, 
    'lr_policy': 'fixed',
    'momentum': 0.9,
    'weight_decay': 0.05,
    'use_gpu': 1,
    'gpu_id': gpu_id,
    'weights_file_prefix': EXP_DIR + 'policy',
    'network_params': {
        'obs_include': agent['obs_include'],
        'sensor_dims': agent['sensor_dims'],
        'dim_hidden': [20, 10, 5],
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
