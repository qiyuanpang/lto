import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np

N = 10000
a=tf.constant(np.random.rand(N,N),shape=[N,N],name='a')
b=tf.constant(np.random.rand(N,N),shape=[N,N],name='b')
c=tf.matmul(a,b)
sess = tf.Session()
print(sess.run(c))
