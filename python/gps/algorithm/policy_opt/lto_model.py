#import tensorflow as tf
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from gps.algorithm.policy_opt.tf_utils import TfMap
import numpy as np
from gps.proto.gps_pb2 import CUR_LOC, PAST_OBJ_VAL_DELTAS, PAST_GRADS, CUR_GRAD, PAST_LOC_DELTAS, ACTION

def init_weights(shape, prev=None, name=None):
    if prev is None:
       #return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)
       return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
    else:
       #return tf.Variable(tf.constant(prev), name=name)
       return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(prev))

def init_bias(shape, prev=None, name=None):
    if prev is None:
       #return tf.Variable(tf.zeros(shape, dtype=tf.float32), name=name)
       return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.constant_initializer())
    else:
       #return tf.Variable(tf.constant(prev), name=name)
       return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(prev))

def batched_matrix_vector_multiply(vector, matrix):
    """ computes x^T A in mini-batches. """
    vector_batch_as_matricies = tf.expand_dims(vector, [1])
    mult_result = tf.matmul(vector_batch_as_matricies, matrix)
    squeezed_result = tf.squeeze(mult_result, [1])
    return squeezed_result

def get_input_layer():
    """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to compute loss."""
    net_input = tf.placeholder("float", [None, None], name='nn_input')  # (N*T) x dO
    action = tf.placeholder('float', [None, None], name='action')       # (N*T) x dU
    precision = tf.placeholder('float', [None, None, None], name='precision') # (N*T) x dU x dU
    return net_input, action, precision

def get_loss_layer(mlp_out, action, precision, batch_size):
    """The loss layer used for the MLP network is obtained through this class."""
    scale_factor = tf.constant(2*batch_size, dtype=tf.float32)
    uP = batched_matrix_vector_multiply(action - mlp_out, precision)
    uPu = tf.reduce_sum(uP*(action - mlp_out))  # this last dot product is then summed, so we just the sum all at once.
    return uPu/scale_factor

def average_gradients(tower_grads):
    average_grads = []
    ##grad_and_vars：((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def fully_connected_tf_network(dim_input, dim_output, batch_size=25, network_config=None):
    
    dim_hidden = network_config['dim_hidden'] + [dim_output]
    n_layers = len(dim_hidden)
    
    nn_input, action, precision = get_input_layer()
    
    weights = []
    biases = []
    in_shape = dim_input
    for layer_step in range(0, n_layers):
        cur_weight = init_weights([in_shape, dim_hidden[layer_step]], name='w_' + str(layer_step))
        cur_bias = init_bias([dim_hidden[layer_step]], name='b_' + str(layer_step))
        in_shape = dim_hidden[layer_step]
        weights.append(cur_weight)
        biases.append(cur_bias)
    
    cur_top = nn_input
    for layer_step in range(0, n_layers):
        if layer_step != n_layers-1:  # final layer has no RELU
            cur_top = tf.nn.relu(tf.matmul(cur_top, weights[layer_step]) + biases[layer_step])
        else:
            cur_top = tf.matmul(cur_top, weights[layer_step]) + biases[layer_step]
    
    mlp_applied = cur_top
    loss_out = get_loss_layer(mlp_out=mlp_applied, action=action, precision=precision, batch_size=batch_size)
    #print(nn_input.get_shape(), action.get_shape())
    return TfMap.init_from_lists([nn_input, action, precision], [mlp_applied], [loss_out])

def fully_connected_tf_network_leaky_relu(dim_input, dim_output, batch_size=25, network_config=None):

    dim_hidden = network_config['dim_hidden'] + [dim_output]
    n_layers = len(dim_hidden)

    nn_input, action, precision = get_input_layer()

    weights = []
    biases = []
    in_shape = dim_input
    for layer_step in range(0, n_layers):
        cur_weight = init_weights([in_shape, dim_hidden[layer_step]], name='w_' + str(layer_step))
        cur_bias = init_bias([dim_hidden[layer_step]], name='b_' + str(layer_step))
        in_shape = dim_hidden[layer_step]
        weights.append(cur_weight)
        biases.append(cur_bias)

    cur_top = nn_input
    for layer_step in range(0, n_layers):
        if layer_step != n_layers-1:  # final layer has no RELU
            #cur_top = tf.nn.relu(tf.matmul(cur_top, weights[layer_step]) + biases[layer_step])
            cur_top = tf.nn.leaky_relu(tf.matmul(cur_top, weights[layer_step]) + biases[layer_step])
        else:
            cur_top = tf.matmul(cur_top, weights[layer_step]) + biases[layer_step]

    mlp_applied = cur_top
    loss_out = get_loss_layer(mlp_out=mlp_applied, action=action, precision=precision, batch_size=batch_size)
    #print(nn_input.get_shape(), action.get_shape())
    return TfMap.init_from_lists([nn_input, action, precision], [mlp_applied], [loss_out])

def fully_connected_tf_network_swish(dim_input, dim_output, batch_size=25, network_config=None):

    dim_hidden = network_config['dim_hidden'] + [dim_output]
    n_layers = len(dim_hidden)

    nn_input, action, precision = get_input_layer()

    weights = []
    biases = []
    in_shape = dim_input
    for layer_step in range(0, n_layers):
        cur_weight = init_weights([in_shape, dim_hidden[layer_step]], name='w_' + str(layer_step))
        cur_bias = init_bias([dim_hidden[layer_step]], name='b_' + str(layer_step))
        in_shape = dim_hidden[layer_step]
        weights.append(cur_weight)
        biases.append(cur_bias)

    cur_top = nn_input
    for layer_step in range(0, n_layers):
        if layer_step != n_layers-1:  # final layer has no RELU
            #cur_top = tf.nn.relu(tf.matmul(cur_top, weights[layer_step]) + biases[layer_step])
            cur_top = tf.nn.swish(tf.matmul(cur_top, weights[layer_step]) + biases[layer_step])
        else:
            cur_top = tf.matmul(cur_top, weights[layer_step]) + biases[layer_step]

    mlp_applied = cur_top
    loss_out = get_loss_layer(mlp_out=mlp_applied, action=action, precision=precision, batch_size=batch_size)
    #print(nn_input.get_shape(), action.get_shape())
    return TfMap.init_from_lists([nn_input, action, precision], [mlp_applied], [loss_out])

def first_derivative_network_leaky_relu(dim_input, dim_output, batch_size=25, network_config=None):


    nn_input, action, precision = get_input_layer()

    weights = []
    biases = []
    in_shape = network_config['history_len']*2 + 1
    dim_hidden = [in_shape] + network_config['dim_hidden'] + [1]
    n_layers = len(dim_hidden) - 1
    param_dim = network_config['param_dim']
    weights_prev = network_config['weights_prev']
    #print('weights=', weights_prev)
    biases_prev = network_config['biases_prev']
    for layer_step in range(n_layers):
        if weights_prev is None:
            w_prev = None
            b_prev = None
        else:
            w_prev = weights_prev[layer_step]
            b_prev = biases_prev[layer_step]
        cur_weight = init_weights([in_shape, dim_hidden[layer_step+1]], w_prev, name='w_'+str(layer_step))
        cur_bias = init_bias([dim_hidden[layer_step+1]], b_prev, name='b_'+str(layer_step))
        in_shape = dim_hidden[layer_step+1]
        weights.append(cur_weight)
        biases.append(cur_bias)
    #print(nn_input.get_shape())
    dim0 = batch_size
    cur_top = nn_input
    for layer_step in range(n_layers):
        top = {}
        #print(weights[layer_step])
        #print('dim_input = ', dim_input)
        for i in range(dim_hidden[layer_step+1]):
            #print('dim0 = ', dim0)
            #top['slice_'+str(i)] = tf.zeros([dim0, param_dim])
            top['slice_'+str(i)] = weights[layer_step][0, i]*cur_top[:, 0:param_dim]
            for j in range(1, dim_hidden[layer_step]):
                #print(i, j, param_dim*j)
                #with tf.Session():
                #print(layer_step, i, j, weights[layer_step][j,i].get_shape())
                #print(top['slice_'+str(i)].get_shape(), tf.slice(cur_top, [0, param_dim*j],[dim0, param_dim]).get_shape())
                #print(cur_top.get_shape(), param_dim*j, param_dim)
                loc = cur_top[:, param_dim*j: param_dim*(j+1)]
                top['slice_'+str(i)] = tf.add(top['slice_'+str(i)], weights[layer_step][j,i]*loc)
            top['slice_'+str(i)] = top['slice_'+str(i)] + biases[layer_step][i]
        cur_top = top['slice_'+str(0)]
        for i in range(1, dim_hidden[layer_step+1]):
            #print(cur_top, top['slice_'+str(i)])
            cur_top = tf.concat([cur_top, top['slice_'+str(i)]], axis = 1)
        if layer_step != n_layers - 1:
            cur_top = tf.nn.leaky_relu(cur_top)
    #print(dim_output, param_dim)
    mlp_applied = cur_top
    loss_out = get_loss_layer(mlp_out=mlp_applied, action=action, precision=precision, batch_size=batch_size)

    return TfMap.init_from_lists([nn_input, action, precision], [mlp_applied], [loss_out])

def first_derivative_network_swish(dim_input, dim_output, batch_size=25, network_config=None):


    nn_input, action, precision = get_input_layer()

    weights = []
    biases = []
    in_shape = network_config['history_len']*2 + 1
    dim_hidden = [in_shape] + network_config['dim_hidden'] + [1]
    n_layers = len(dim_hidden) - 1
    param_dim = network_config['param_dim']
    weights_prev = network_config['weights_prev']
    #print('weights=', weights_prev)
    biases_prev = network_config['biases_prev']
    for layer_step in range(n_layers):
        if weights_prev is None:
            w_prev = None
            b_prev = None
        else:
            w_prev = weights_prev[layer_step]
            b_prev = biases_prev[layer_step]
        cur_weight = init_weights([in_shape, dim_hidden[layer_step+1]], w_prev, name='w_'+str(layer_step))
        cur_bias = init_bias([dim_hidden[layer_step+1]], b_prev, name='b_'+str(layer_step))
        in_shape = dim_hidden[layer_step+1]
        weights.append(cur_weight)
        biases.append(cur_bias)
    #print(nn_input.get_shape())
    dim0 = batch_size
    cur_top = nn_input
    for layer_step in range(n_layers):
        top = {}
        #print(weights[layer_step])
        #print('dim_input = ', dim_input)
        for i in range(dim_hidden[layer_step+1]):
            #print('dim0 = ', dim0)
            #top['slice_'+str(i)] = tf.zeros([dim0, param_dim])
            top['slice_'+str(i)] = weights[layer_step][0, i]*cur_top[:, 0:param_dim]
            for j in range(1, dim_hidden[layer_step]):
                #print(i, j, param_dim*j)
                #with tf.Session():
                #print(layer_step, i, j, weights[layer_step][j,i].get_shape())
                #print(top['slice_'+str(i)].get_shape(), tf.slice(cur_top, [0, param_dim*j],[dim0, param_dim]).get_shape())
                #print(cur_top.get_shape(), param_dim*j, param_dim)
                loc = cur_top[:, param_dim*j: param_dim*(j+1)]
                top['slice_'+str(i)] = tf.add(top['slice_'+str(i)], weights[layer_step][j,i]*loc)
            top['slice_'+str(i)] = top['slice_'+str(i)] + biases[layer_step][i]
        cur_top = top['slice_'+str(0)]
        for i in range(1, dim_hidden[layer_step+1]):
            #print(cur_top, top['slice_'+str(i)])
            cur_top = tf.concat([cur_top, top['slice_'+str(i)]], axis = 1)
        if layer_step != n_layers - 1:
            cur_top = tf.nn.swish(cur_top)
    #print(dim_output, param_dim)
    mlp_applied = cur_top
    loss_out = get_loss_layer(mlp_out=mlp_applied, action=action, precision=precision, batch_size=batch_size)

    return TfMap.init_from_lists([nn_input, action, precision], [mlp_applied], [loss_out])


def first_derivative_network(dim_input, dim_output, batch_size=25, network_config=None):


    nn_input, action, precision = get_input_layer()
    gpu_ids = network_config['gpu_ids']
    solver_type = network_config['solver_type']
    with tf.variable_scope("train"):
        weights = []
        biases = []
        in_shape = network_config['history_len']*2 + 1
        dim_hidden = [in_shape] + network_config['dim_hidden'] + [1]
        n_layers = len(dim_hidden) - 1
        param_dim = network_config['param_dim']
        weights_prev = network_config['weights_prev']
        #print('weights=', weights_prev)
        biases_prev = network_config['biases_prev']
        for layer_step in range(n_layers):
            if weights_prev is None:
                w_prev = None
                b_prev = None
            else:
                w_prev = weights_prev[layer_step]
                b_prev = biases_prev[layer_step]
            cur_weight = init_weights([in_shape, dim_hidden[layer_step+1]], w_prev, name='w_'+str(layer_step))
            cur_bias = init_bias([dim_hidden[layer_step+1]], b_prev, name='b_'+str(layer_step))
            in_shape = dim_hidden[layer_step+1]
            weights.append(cur_weight)
            biases.append(cur_bias)
    #print(nn_input.get_shape())
    dim0 = batch_size
    def my_net(nn_input, n_layers, dim_hidden, param_dim, weights, biases):
        cur_top = nn_input
        for layer_step in range(n_layers):
            top = {}
            #print(weights[layer_step])
            #print('dim_input = ', dim_input)
            for i in range(dim_hidden[layer_step+1]):
                #print('dim0 = ', dim0)
                #top['slice_'+str(i)] = tf.zeros([dim0, param_dim])
                top['slice_'+str(i)] = weights[layer_step][0, i]*cur_top[:, 0:param_dim]
                for j in range(1, dim_hidden[layer_step]):
                    #print(i, j, param_dim*j)
                    #with tf.Session():
                    #print(layer_step, i, j, weights[layer_step][j,i].get_shape())
                    #print(top['slice_'+str(i)].get_shape(), tf.slice(cur_top, [0, param_dim*j],[dim0, param_dim]).get_shape())
                    #print(cur_top.get_shape(), param_dim*j, param_dim)
                    loc = cur_top[:, param_dim*j: param_dim*(j+1)]
                    top['slice_'+str(i)] = tf.add(top['slice_'+str(i)], weights[layer_step][j,i]*loc)
                top['slice_'+str(i)] = top['slice_'+str(i)] + biases[layer_step][i]
            cur_top = top['slice_'+str(0)]
            for i in range(1, dim_hidden[layer_step+1]):
                #print(cur_top, top['slice_'+str(i)])
                cur_top = tf.concat([cur_top, top['slice_'+str(i)]], axis = 1)
            if layer_step != n_layers - 1:
                cur_top = tf.nn.relu(cur_top)
        #print(dim_output, param_dim)
        mlp_applied = cur_top
        return mlp_applied

    with tf.variable_scope("train", reuse=True):
        mlp_applied = my_net(nn_input, n_layers, dim_hidden, param_dim, weights, biases)
        loss_out = get_loss_layer(mlp_out=mlp_applied, action=action, precision=precision, batch_size=batch_size)

    nn_input_splits = tf.split(nn_input, num_or_size_splits=len(gpu_ids), axis=0)
    action_splits = tf.split(action, num_or_size_splits=len(gpu_ids), axis=0)
    precision_splits = tf.split(precision, num_or_size_splits=len(gpu_ids), axis=0)
    tower_grads = []
    tower_loss = []
    mlp_applied_all = []
    optimizer = get_optimizer(solver_type, network_config)
    with tf.variable_scope("train", reuse=True):
        for i in range(len(gpu_ids)):
            gpu_id = gpu_ids[i]
            with tf.device('/gpu:%d' % gpu_id):
                mlp_applied_each = my_net(nn_input_splits[i], n_layers, dim_hidden, param_dim, weights, biases)
                loss_out_each = get_loss_layer(mlp_out=mlp_applied_each, action=action_splits[i], precision=precision_splits[i], batch_size=int(batch_size/len(gpu_ids)))
                tf.summary.scalar('loss_out_gpu%d' % gpu_id, loss_out_each)
                grads = optimizer.compute_gradients(loss_out_each)
                tower_grads.append(grads)
                tower_loss.append(loss_out_each)
                mlp_applied_all.append(mlp_applied_each)
    avg_tower_loss = tf.reduce_mean(tower_loss, axis=0)
    tf.summary.scalar('avg_tower_loss', avg_tower_loss)
    grads_avg = average_gradients(tower_grads)
    summary_op = tf.summary.merge_all()
    solver_op = optimizer.apply_gradients(grads_avg)
    mlp_applied_4prob = mlp_applied_all[0]
    for i in range(1, len(gpu_ids)):
        mlp_applied_4prob = tf.concat([mlp_applied_4prob, mlp_applied_all[i]], axis=0)


    return TfMap.init_from_lists([nn_input, action, precision], [mlp_applied], [loss_out]), solver_op, summary_op, avg_tower_loss, mlp_applied_4prob

def get_optimizer(solver_type, network_config):
    solver_string = solver_type.lower()
    if solver_string == 'adam':
        return tf.train.AdamOptimizer(learning_rate=network_config['lr'],beta1=network_config['momentum'],beta2=network_config['momentum2'],epsilon=network_config['epsilon'])
    elif solver_string == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate=network_config['lr'],decay=network_config['momentum'])
    elif solver_string == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate=network_config['lr'],momentum=network_config['momentum'])
    elif solver_string == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate=network_config['lr'],initial_accumulator_value=network_config['momentum'])
    elif solver_string == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=network_config['lr'])
    else:
        raise NotImplementedError("Please select a valid optimizer.")

def fcn4rnn(nn_input, dims, weights, biases, dim_hidden, flag):
    cur_top = nn_input
    n_layers = len(dim_hidden) - 1
    for layer_step in range(n_layers):
        top = dict()
        for i in range(dim_hidden[layer_step+1]):
            top[i] = weights[layer_step][0, i]*cur_top[:, 0:dims]
            for j in range(1, dim_hidden[layer_step]):
                loc = cur_top[:, dims*j:dims*(j+1)]
                #if flag == 'outer':
                #    print(tf.shape(top[i]))
                #    print(tf.shape(loc))
                #    print(weights[layer_step][j, i])
                top[i] = tf.add(top[i], weights[layer_step][j,i]*loc)
            top[i] = top[i] + biases[layer_step][i]
        cur_top = top[0]
        for i in range(1, dim_hidden[layer_step+1]):
            cur_top = tf.concat([cur_top, top[i]], axis=1)
        if flag == 'inner':
            cur_top = tf.nn.tanh(cur_top)
        elif flag == 'outer':
            if layer_step != n_layers - 1:
                cur_top = tf.nn.relu(cur_top)
    return cur_top
            

def recurrent_neural_network_multilayers(dim_input, dim_output, batch_size=25, network_config=None):
    
    nn_input, action, precision = get_input_layer()
    weights = []
    biases = []
    T = network_config['history_len']
    out_shape = 3
    in_shape = out_shape + 2
    dim_hidden = [in_shape] + network_config['dim_hidden'] + [out_shape]
    n_layers = len(dim_hidden) - 1
    param_dim = network_config['param_dim']
    weights_prev = network_config['weights_prev']
    biases_prev = network_config['biases_prev']    
    in_shape1 = in_shape
    for layer_step in range(n_layers):
        if weights_prev is None:
            w_prev = None
            b_prev = None
        else:
            w_prev = weights_prev[layer_step]
            b_prev = biases_prev[layer_step]
        cur_weight = init_weights([in_shape1, dim_hidden[layer_step+1]], w_prev, name='w_'+str(layer_step))
        cur_bias = init_bias([dim_hidden[layer_step+1]], b_prev, name='b_'+str(layer_step))
        in_shape1 = dim_hidden[layer_step+1]
        weights.append(cur_weight)
        biases.append(cur_bias)
    
    weights_output = []
    biases_output = []
    dim_hidden_output = [out_shape] + network_config['dim_hidden_output'] + [1]
    n_layers_output = len(dim_hidden_output)-1
    weights_prev_output = network_config['weights_prev_output']
    biases_prev_output = network_config['biases_prev_output']
    out_shape1 = out_shape
    for layer_step in range(n_layers_output):
        if weights_prev_output is None:
            w_prev = None
            b_prev = None
        else:
            w_prev = weights_prev_output[layer_step]
            b_prev = biases_prev_output[layer_step]
        cur_weight = init_weights([out_shape1, dim_hidden_output[layer_step+1]], w_prev, name='w_o_'+str(layer_step))
        cur_bias = init_bias([dim_hidden_output[layer_step+1]], b_prev, name='b_o_'+str(layer_step))
        out_shape1 = dim_hidden_output[layer_step+1]
        weights_output.append(cur_weight)
        biases_output.append(cur_bias)
        
    h = nn_input[:, 0:out_shape*param_dim]*0.0
    for t in range(T):
        grad_t = nn_input[:, t*param_dim:(t+1)*param_dim]
        loc_t = nn_input[:, (T+t)*param_dim:(T+t+1)*param_dim]
        nn_input_t = tf.concat([grad_t, loc_t], axis=1)
        nn_input_t = tf.concat([nn_input_t, h], axis=1)
        h = fcn4rnn(nn_input_t, param_dim, weights, biases, dim_hidden, 'inner')
    grad_cur = nn_input[:, (2*T)*param_dim:(2*T+1)*param_dim]
    loc_cur = nn_input[:, (2*T+1)*param_dim:(2*T+2)*param_dim]
    nn_input_cur = tf.concat([grad_cur, loc_cur], axis=1)
    nn_input_cur = tf.concat([nn_input_cur, h], axis=1)
    h = fcn4rnn(nn_input_cur, param_dim, weights, biases, dim_hidden, 'inner')  
    
    y = fcn4rnn(h, param_dim, weights_output, biases_output, dim_hidden_output, 'outer')
    mlp_applied = y
    loss_out = get_loss_layer(mlp_out=mlp_applied, action=action, precision=precision, batch_size=batch_size)

    return TfMap.init_from_lists([nn_input, action, precision], [mlp_applied], [loss_out])
