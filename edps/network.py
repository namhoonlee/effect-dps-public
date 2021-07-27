import tensorflow as tf

from functools import reduce
from helpers import static_size


def load_network(**kwargs):
    arch = kwargs.pop('arch')
    networks = {
        'simple-cnn': lambda: SimpleCNN(**kwargs),
        'resnet-8': lambda: ResNetNB(**kwargs),
    }
    return networks[arch]()


class Network(object):
    def __init__(self, datasource):
        self.datasource = datasource
        if self.datasource == 'mnist':
            self.input_dims = [28, 28, 1]
            self.output_dims = 10
        elif self.datasource == 'fashion-mnist':
            self.input_dims = [28, 28, 1]
            self.output_dims = 10
        elif self.datasource == 'cifar-10':
            self.input_dims = [32, 32, 3]
            self.output_dims = 10
        else:
            raise NotImplementedError

    def construct_inputs(self):
        return {
            'image': tf.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, trainable, scope):
        params = {
            'dtype': tf.float32,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        weights = {}
        with tf.variable_scope(scope):
            for k, v in self.weights_shape.items():
                if len(v) == 1: # b
                    params.update({'initializer': tf.zeros_initializer()})
                else: # w
                    params.update({'initializer': tf.variance_scaling_initializer()})
                weights[k] = tf.get_variable(k, v, **params)
        return weights

    def construct_network(self):
        self.inputs = self.construct_inputs()
        self.weights_shape, self.states_shape = self.set_weights_shape()
        self.weights = self.construct_weights(True, 'net')
        self.num_params = sum([static_size(v) for v in self.weights.values()])


class SimpleCNN(Network):
    def __init__(self, datasource):
        super(SimpleCNN, self).__init__(datasource)
        self.name = 'simple-cnn'
        self.construct_network()

    def set_weights_shape(self):
        weights_shape = {
            'w1': [5, 5, 1, 32],
            'w2': [5, 5, 32, 64],
            'w3': [1024, self.output_dims],
            'b1': [32],
            'b2': [64],
            'b3': [self.output_dims],
        }
        return weights_shape, None

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        inputs = tf.nn.conv2d(inputs, weights['w1'], [1, 1, 1, 1], 'VALID') + weights['b1']
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        inputs = tf.nn.conv2d(inputs, weights['w2'], [1, 1, 1, 1], 'VALID') + weights['b2']
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        inputs = tf.reshape(inputs, [-1, reduce(lambda x, y: x*y, inputs.shape.as_list()[1:])])
        inputs = tf.matmul(inputs, weights['w3']) + weights['b3']
        return inputs


class ResNetNB(Network):
    def __init__(self, datasource):
        super(ResNetNB, self).__init__(datasource)
        self.name = 'resnet-8'
        self.num_filters = 16
        self.k = 1
        self.block_sizes = [1, 1, 1]
        self.num_block_layers = len(self.block_sizes)
        self.construct_network()

    def set_weights_shape(self):
        weights_shape = {}
        cnt = 1
        weights_shape.update({
            'w1': [3, 3, 3, self.num_filters],
            'b1': [self.num_filters],
        })
        nChIn = self.num_filters
        for l in range(self.num_block_layers):
            nChOut = self.num_filters * (2**l) * self.k
            for i in range(2 * self.block_sizes[l]):
                cnt += 1
                weights_shape.update({
                    'w{}'.format(cnt): [3, 3, nChIn if i == 0 else nChOut, nChOut],
                    'b{}'.format(cnt): [nChOut],
                })
            nChIn = nChOut
        cnt += 1
        weights_shape.update({
            'w{}'.format(cnt): [nChOut, self.output_dims],
            'b{}'.format(cnt): [self.output_dims],
        })
        return weights_shape, None

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        def _res_block(inputs, filt1, filt2, filt3=None, zeropad=False, k=1, st=1):
            shortcut = inputs
            inputs = tf.nn.relu(inputs)
            if filt3 is not None:
                shortcut = tf.nn.conv2d(inputs, filt3['w'], [1, st, st, 1], 'SAME') + filt3['b']
            elif zeropad:
                shortcut = tf.nn.avg_pool(inputs, [1, st, st, 1], [1, st, st, 1], 'SAME')
                shortcut = tf.concat([shortcut] + [tf.zeros_like(shortcut)]*(k-1), -1)
            else:
                pass
            inputs = tf.nn.conv2d(inputs, filt1['w'], [1, st, st, 1], 'SAME') + filt1['b']
            inputs = tf.nn.relu(inputs)
            inputs = tf.nn.conv2d(inputs, filt2['w'], [1, 1, 1, 1], 'SAME') + filt2['b']
            return inputs + shortcut

        cnt = 1
        inputs = tf.nn.conv2d(inputs, weights['w1'], [1, 1, 1, 1], 'SAME') + weights['b1']
        for l in range(self.num_block_layers):
            dims = weights['w{}'.format(cnt+1)].shape.as_list()
            if dims[2] != dims[3]:
                inputs = _res_block(inputs,
                    {'w': weights['w{}'.format(cnt+1)], 'b': weights['b{}'.format(cnt+1)]},
                    {'w': weights['w{}'.format(cnt+2)], 'b': weights['b{}'.format(cnt+2)]},
                    zeropad=True,
                    k=self.k if l == 0 else 2,
                    st=1 if l == 0 else 2,
                )
            else:
                inputs = _res_block(inputs,
                    {'w': weights['w{}'.format(cnt+1)], 'b': weights['b{}'.format(cnt+1)]},
                    {'w': weights['w{}'.format(cnt+2)], 'b': weights['b{}'.format(cnt+2)]},
                )
            cnt += 2
            for _ in range(1, self.block_sizes[l]):
                inputs = _res_block(inputs,
                    {'w': weights['w{}'.format(cnt+1)], 'b': weights['b{}'.format(cnt+1)]},
                    {'w': weights['w{}'.format(cnt+2)], 'b': weights['b{}'.format(cnt+2)]},
                )
                cnt += 2
        inputs = tf.nn.relu(inputs)
        cnt += 1
        assert inputs.shape.as_list()[1] == 8
        inputs = tf.reduce_mean(inputs, [1, 2])
        inputs = tf.matmul(inputs, weights['w{}'.format(cnt)]) + weights['b{}'.format(cnt)]
        return inputs
