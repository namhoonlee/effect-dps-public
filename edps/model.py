import tensorflow as tf
import functools
import numpy as np

from functools import reduce
from network import load_network

dtype = tf.float32

class Model(object):
    def __init__(self,
                 datasource,
                 arch,
                 target_sparsity,
                 optimizer,
                 learning_rate,
                 decay_type,
                 decay_steps,
                 end_learning_rate_factor,
                 momentum,
                 check_lipschitz,
                 **kwargs):
        self.datasource = datasource
        self.arch = arch
        self.target_sparsity = target_sparsity
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.decay_type = decay_type
        self.decay_steps = decay_steps
        self.end_learning_rate_factor = end_learning_rate_factor
        self.momentum = momentum
        self.check_lipschitz = check_lipschitz

    def construct_model(self, inputs=None):
        # Base-learner
        self.net = net = load_network(**{'arch': self.arch, 'datasource': self.datasource})

        # Input nodes
        self.inputs = net.inputs
        self.compress = tf.placeholder_with_default(False, [])
        self.accumulate_g = tf.placeholder_with_default(False, [])
        self.is_train = tf.placeholder_with_default(False, [])

        # For convenience
        prn_keys = [k for p in ['w', 'b'] for k in net.weights.keys() if p in k]
        var_no_train = functools.partial(tf.Variable, trainable=False, dtype=dtype)
        self.weights = weights = net.weights

        # Prune
        mask_init = {k: var_no_train(tf.ones(weights[k].shape)) for k in prn_keys}
        mask_prev = {k: var_no_train(tf.ones(weights[k].shape)) for k in prn_keys}
        g_mmt_prev = {k: var_no_train(tf.zeros(weights[k].shape)) for k in prn_keys}
        cs_prev = {k: var_no_train(tf.zeros(weights[k].shape)) for k in prn_keys}
        def update_g_mmt():
            w_mask = apply_mask(weights, mask_init)
            logits = net.forward_pass(w_mask, self.inputs['image'],
                self.is_train, trainable=False)
            loss = tf.reduce_mean(compute_loss(self.inputs['label'], logits))
            grads = tf.gradients(loss, [mask_init[k] for k in prn_keys])
            gradients = dict(zip(prn_keys, grads))
            g_mmt = {k: g_mmt_prev[k] + gradients[k] for k in prn_keys}
            return g_mmt
        g_mmt = tf.cond(self.accumulate_g, lambda: update_g_mmt(), lambda: g_mmt_prev)
        def get_sparse_mask():
            cs = normalize_dict({k: tf.abs(g_mmt[k]) for k in prn_keys})
            return (create_sparse_mask(cs, self.target_sparsity), cs)
        with tf.control_dependencies([tf.assign(g_mmt_prev[k], g_mmt[k]) for k in prn_keys]):
            mask, cs = tf.cond(self.compress,
                lambda: get_sparse_mask(), lambda: (mask_prev, cs_prev))
        with tf.control_dependencies([tf.assign(mask_prev[k], v) for k,v in mask.items()]):
            w_final = apply_mask(weights, mask)

        # Sparsity
        self.sparsity = compute_sparsity(w_final, prn_keys)

        # Forward pass
        logits = net.forward_pass(w_final, self.inputs['image'], self.is_train)

        # Optimization
        loss_opt = tf.reduce_mean(compute_loss(self.inputs['label'], logits))
        optim, learning_rate, global_step = prepare_optimization(
            self.optimizer, self.learning_rate, self.decay_type,
            self.decay_steps, self.end_learning_rate_factor, self.momentum)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optim.minimize(loss_opt, global_step=global_step)

        # Measure Lipschitz smoothness
        if self.check_lipschitz:
            def _assign_weights(w_to, w_from):
                with tf.control_dependencies([tf.assign(w_to[k], w_from[k]) for k in prn_keys]):
                    return {k: tf.identity(w_to[k]) for k in prn_keys}

            self.grad = tf.gradients(loss_opt, [weights[k] for k in prn_keys])

            # many auxiliary graph nodes
            self.update_w_prev = tf.placeholder_with_default(False, [])
            w_prev = {k: var_no_train(tf.zeros(weights[k].shape)) for k in prn_keys}
            w_prev = tf.cond(self.update_w_prev,
                lambda: _assign_weights(w_prev, weights), lambda: w_prev)
            self.w_prev = w_prev

            self.update_w_copy = tf.placeholder_with_default(False, [])
            w_copy = {k: var_no_train(tf.zeros(weights[k].shape)) for k in prn_keys}
            w_copy = tf.cond(self.update_w_copy,
                lambda: _assign_weights(w_copy, weights), lambda: w_copy)
            self.w_copy = w_copy

            self.gamma = tf.placeholder(dtype, [])
            w_new = {k: ((1-self.gamma) * w_prev[k]) + (self.gamma * w_copy[k]) for k in prn_keys}

            self.update_weights = tf.placeholder_with_default(False, [])
            w_new = tf.cond(self.update_weights,
                lambda: _assign_weights(weights, w_new), lambda: w_new)
            self.w_new = w_new

            with tf.control_dependencies([tf.assign(weights[k], w_copy[k]) for k in prn_keys]):
                self.w_direction = {k: weights[k] - w_prev[k] for k in prn_keys}
                w_direction_vec, _ = vectorize_dict(self.w_direction)
                self.denom = tf.norm(self.gamma * w_direction_vec)

        # Outputs
        output_class = tf.argmax(logits, axis=1, output_type=tf.int32)
        output_correct_prediction = tf.equal(self.inputs['label'], output_class)
        output_accuracy_individual = tf.cast(output_correct_prediction, dtype)
        output_accuracy = tf.reduce_mean(output_accuracy_individual)
        self.outputs = {
            'los': loss_opt,
            'acc': output_accuracy,
            'acc_individual': output_accuracy_individual,
        }


def compute_loss(labels, logits, fake_uniform=False):
    ''' Computes cross-entropy loss.
    '''
    assert len(labels.shape)+1 == len(logits.shape)
    num_classes = logits.shape.as_list()[-1]
    labels = tf.one_hot(labels, num_classes, dtype=dtype)
    if fake_uniform:
        labels = tf.ones_like(logits) / num_classes
    return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

def get_optimizer(optimizer, learning_rate, momentum):
    if optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    elif optimizer == 'nesterov':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
    else:
        raise NotImplementedError
    return optimizer

def prepare_optimization(optimizer, learning_rate, decay_type,
        decay_steps=None, end_learning_rate_factor=None, momentum=0.9):
    # variable to count traininig steps
    global_step = tf.Variable(0, trainable=False)
    # learning rate schedule
    if decay_type == 'constant':
        learning_rate = tf.constant(learning_rate)
    elif decay_type == 'linear': # Mainly used for batch science.
        end_learning_rate = end_learning_rate_factor * learning_rate
        learning_rate = tf.train.polynomial_decay(
            learning_rate, global_step, decay_steps, end_learning_rate, power=1.0, cycle=False)
    else:
        raise NotImplementedError
    # optimizer
    optim = get_optimizer(optimizer, learning_rate, momentum)
    return optim, learning_rate, global_step

def sum_lp_loss(w, order):
    '''
        order | return
        0     | sum_i ||w_i||_0 = sum_i deltafunc_{w_i}
        1     | sum_i |w_i|
        2     | sum_i w_i^2
    '''
    assert isinstance(w, dict)
    func = {
        0: lambda x: tf.cast(tf.greater(tf.abs(x), 0.0), dtype),
        1: tf.abs,
        2: tf.square,
    }
    return tf.reduce_sum([tf.reduce_sum(func[order](v)) for v in w.values()])

def vectorize_dict(x, sortkeys=None):
    assert isinstance(x, dict)
    if sortkeys is None:
        sortkeys = x.keys()
    def restore(v, x_shape, sortkeys):
        # v splits for each key
        split_sizes = []
        for key in sortkeys:
            split_sizes.append(reduce(lambda x, y: x*y, x_shape[key]))
        v_splits = tf.split(v, num_or_size_splits=split_sizes)
        # x restore
        x_restore = {}
        for i, key in enumerate(sortkeys):
            x_restore.update({key: tf.reshape(v_splits[i], x_shape[key])})
        return x_restore
    # vectorized dictionary
    x_vec = tf.concat([tf.reshape(x[k], [-1]) for k in sortkeys], axis=0)
    # restore function
    x_shape = {k: x[k].shape.as_list() for k in sortkeys}
    restore_fn = functools.partial(restore, x_shape=x_shape, sortkeys=sortkeys)
    return x_vec, restore_fn

def normalize_dict(x):
    '''
    Normalize the values in dictionary.
    Use to normalize connection sensitivity.
    '''
    x_v, restore_fn = vectorize_dict(x)
    x_v_norm = tf.divide(x_v, tf.reduce_sum(x_v))
    x_norm = restore_fn(x_v_norm)
    return x_norm

def compute_sparsity(weights, target_keys):
    assert isinstance(weights, dict)
    w = {k: weights[k] for k in target_keys}
    w_v, _ = vectorize_dict(w)
    sparsity = tf.nn.zero_fraction(w_v)
    return sparsity

def create_sparse_mask(mask, target_sparsity, soft=False):
    def threshold_vec(vec, target_sparsity, soft):
        num_params = vec.shape.as_list()[0]
        kappa = int(round(num_params * (1. - target_sparsity)))
        if soft:
            topk, ind = tf.nn.top_k(vec, k=kappa+1, sorted=True)
            mask_sparse_v = tf.to_float(tf.greater(vec, topk[-1]))
        else:
            topk, ind = tf.nn.top_k(vec, k=kappa, sorted=True)
            mask_sparse_v = tf.sparse_to_dense(ind, tf.shape(vec),
                tf.ones_like(ind, dtype=dtype), validate_indices=False)
        return mask_sparse_v
    if isinstance(mask, dict):
        mask_v, restore_fn = vectorize_dict(mask)
        mask_sparse_v = threshold_vec(mask_v, target_sparsity, soft)
        return restore_fn(mask_sparse_v)
    else:
        return threshold_vec(mask, target_sparsity, soft)

def apply_mask(weights, mask):
    all_keys = weights.keys()
    target_keys = mask.keys()
    remain_keys = list(set(all_keys) - set(target_keys))
    w_sparse = {k: mask[k] * weights[k] for k in target_keys}
    w_sparse.update({k: weights[k] for k in remain_keys})
    return w_sparse
