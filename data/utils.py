import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tflearn.initializations import truncated_normal 
from tflearn.activations import relu

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def row_normalize(matrix, substract_self_loop):
    a_matrix = matrix.copy()
    if substract_self_loop == True:
        np.fill_diagonal(a_matrix,0)
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1)+1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix

def set_diagonal(matrix, set_value=0):
    a_matrix = matrix.copy()
    np.fill_diagonal(a_matrix,set_value)
    return a_matrix


def sample(matrix):
    pos_idx = np.array(np.where(matrix==1.0)).astype('int32').T
    neg_idx = np.array(np.where(matrix==0.0)).astype('int32').T
    pos_idx_c = np.random.choice(np.arange(len(pos_idx)), len(pos_idx), replace=True)
    neg_idx_c = np.random.choice(np.arange(len(neg_idx)), len(pos_idx), replace=True)
    return pos_idx[pos_idx_c], neg_idx[neg_idx_c]

def rank_loss(y_pred, pos_idx, neg_idx):
    sampled_pos_pred = tf.gather_nd(y_pred, pos_idx)
    sampled_neg_pred = tf.gather_nd(y_pred, neg_idx)
    pred_logits = sampled_pos_pred - sampled_neg_pred
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pred_logits), logits=pred_logits))
    return loss 
    
def square_loss(y_pred, y_label):
    loss = tf.reduce_sum(tf.square(y_label-y_pred))
    return loss


#def weight_variable(shape):
#    return tf.Variable(tf.contrib.layers.xavier_initializer()((shape[0], shape[1])))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)

def bias_one_variable(shape):
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)


def a_layer(x,units):
    W = weight_variable([x.get_shape().as_list()[1],units])
    b = bias_variable([units])
    tf.add_to_collection('eucl_vars', W)
    tf.add_to_collection('eucl_vars', b)
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W))
    return relu(tf.matmul(x, W) + b)


def bi_layer(x0,x1,sym,dim_pred):
    if sym == False:
        W0p = weight_variable([x0.get_shape().as_list()[1],dim_pred])
        W1p = weight_variable([x1.get_shape().as_list()[1],dim_pred])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0p))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W1p))
        return tf.matmul(tf.matmul(x0, W0p), 
                            tf.matmul(x1, W1p),transpose_b=True)
    else:
        W0p = weight_variable([x0.get_shape().as_list()[1],dim_pred])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0p))
        return tf.matmul(tf.matmul(x0, W0p), 
                            tf.matmul(x1, W0p),transpose_b=True)


def euclidean_dist_sq(A,B):
    Ar = tf.reshape(tf.reduce_sum(A*A, 1), [-1,1])
    Br = tf.reshape(tf.reduce_sum(B*B, 1), [1,-1])
    D = Ar - 2*tf.matmul(A, tf.transpose(B)) + Br
    return D
