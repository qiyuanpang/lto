import pickle
import os
import uuid

import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()


from gps.algorithm.policy.policy import Policy


class TfPolicy(Policy):
    """
    A neural network policy implemented in TensorFlow. The network output is
    taken to be the mean, and Gaussian noise is added on top of it.
    U = net.forward(obs) + noise, where noise ~ N(0, diag(var))
    Args:
        obs_tensor: tensor representing tf observation. Used in feed dict for forward pass.
        act_op: tf op to execute the forward pass. Use sess.run on this op.
        var: Du-dimensional noise variance vector.
        sess: tf session.
        device_string: tf device string for running on either gpu or cpu.
    """
    def __init__(self, dU, obs_tensor, act_op, var, sess, device_string):
        Policy.__init__(self)
        self.dU = dU
        self.obs_tensor = obs_tensor
        self.act_op = act_op
        self.sess = sess
        self.device_string = device_string
        self.chol_pol_covar = np.diag(np.sqrt(var))
        self.scale = None  # must be set from elsewhere based on observations
        self.bias = None

    def act(self, x, obs, t, noise, usescale=True):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """
        #print('x = ', type(x), len(x))
        #print('obs = ', type(x), len(x))
        # Normalize obs.
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        #print(type(self.scale), type(self.bias))
        #print(self.scale.shape, self.bias.shape)
        #print(self.bias)
        if usescale:
            obs = obs.dot(self.scale) + self.bias
        #print('tf_policy:', self.device_string)
        with tf.device(self.device_string):
            action_mean = self.sess.run(self.act_op, feed_dict={self.obs_tensor: obs})
        if noise is None:
            u = action_mean
        else:
            u = action_mean + self.chol_pol_covar.T.dot(noise)
        #print('action = ', u[0])
        return u[0]  # the DAG computations are batched by default, but we use batch size 1.


    def pickle_policy(self, deg_obs, deg_action, checkpoint_path, goal_state=None, should_hash=False):
        """
        We can save just the policy if we are only interested in running forward at a later point
        without needing a policy optimization class. Useful for debugging and deploying.
        """
        if should_hash is True:
            hash_str = str(uuid.uuid4())
            checkpoint_path += hash_str
        pickled_pol = {'deg_obs': deg_obs, 'deg_action': deg_action, 'chol_pol_covar': self.chol_pol_covar,
                       'checkpoint_path_tf': checkpoint_path + '_tf_data.ckpt', 'scale': self.scale, 'bias': self.bias,
                       'device_string': self.device_string, 'goal_state': goal_state}
        if os.path.isfile(checkpoint_path+'.pkl'):
            os.remove(checkpoint_path+'.pkl')
        pickle.dump(pickled_pol, open(checkpoint_path + '.pkl', "wb"))
        ckpt = checkpoint_path + '_tf_data.ckpt'
        if os.path.isfile(ckpt+'.index'):
            os.remove(ckpt+'.index')
        if os.path.isfile(ckpt+'.data-00000-of-00001'):
            os.remove(ckpt+'.data-00000-of-00001')
        if os.path.isfile(ckpt+'.meta'):
            os.remove(ckpt+'.meta')
        saver = tf.train.Saver()
        saver.save(self.sess, checkpoint_path + '_tf_data.ckpt')

    @classmethod
    def load_policy(cls, policy_dict_path, tf_generator, network_config=None):
        """
        For when we only need to load a policy for the forward pass. For instance, to run on the robot from
        a checkpointed policy.
        """
        from tensorflow.python.framework import ops
        ops.reset_default_graph()  # we need to destroy the default graph before re_init or checkpoint won't restore.
        pol_dict = pickle.load(open(policy_dict_path, "rb"), encoding='latin1')
        if 'deg_obs' in network_config:
            pol_dict['deg_obs'] = network_config['deg_obs']
        if 'deg_action' in network_config:
            pol_dict['deg_action'] = network_config['deg_action']
        
        tf_map,_,_,_,_ = tf_generator(dim_input=pol_dict['deg_obs'], dim_output=pol_dict['deg_action'],
                              batch_size=1, network_config=network_config)

        #sess = tf.Session()
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True))
        #init_op = tf.initialize_all_variables()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver = tf.train.Saver()
        check_file = '/'.join(str.split(policy_dict_path, '/')[:-1]) + '/' + str.split(pol_dict['checkpoint_path_tf'], '/')[-1]
       
        #print('checking file path:', check_file)
        saver.restore(sess, check_file)

        device_string = pol_dict['device_string']
        #keys = [[key, pol_dict[key].shape] for key in pol_dict.keys()]
        #print('tf_map = ', tf_map.get_output_op())
        #print('policy = ', pol_dict['bias'].shape, pol_dict['scale'].shape, pol_dict['chol_pol_covar'].shape)
        #print(pol_dict) 
        print(pol_dict['deg_action'], tf_map.get_input_tensor(), tf_map.get_output_op())
        cls_init = cls(pol_dict['deg_action'], tf_map.get_input_tensor(), tf_map.get_output_op(), np.zeros((1,)),
                       sess, device_string)
        cls_init.chol_pol_covar = pol_dict['chol_pol_covar']
        cls_init.scale = pol_dict['scale']
        cls_init.bias = pol_dict['bias']
        return cls_init

