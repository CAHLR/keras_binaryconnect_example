import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,BatchNormalization
from binary_ops import binary_tanh as binary_tanh_op
from keras import backend as K
from keras.layers import InputSpec, Layer, Conv2D
from keras import constraints
import tensorflow as tf
from keras import initializers
from keras.layers import Input,add
from keras.models import Model
from tensorflow.python.framework import tensor_shape, ops
from tensorflow.python.ops import standard_ops, nn, variable_scope, math_ops, control_flow_ops
from tensorflow.python.eager import context
from tensorflow.python.training import optimizer, training_ops
from binary_ops import binarize
def binary_tanh(x):
    return binary_tanh_op(x)
H = 1.
kernel_lr_multiplier = 'Glorot'
use_bias =True
input_dim =2


class Clip(constraints.Constraint):
    def __init__(self, min_value, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        if not self.max_value:
            self.max_value = -self.min_value
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value

    def __call__(self, p):
        return K.clip(p, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value,
                "max_value": self.max_value}


class BinaryDense(Dense):
    ''' Binarized Dense layer
    References:
    "BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]
    '''

    def __init__(self, units, H=1., binary=True, kernel_lr_multiplier='Glorot', bias_lr_multiplier=None, **kwargs):
        super(BinaryDense, self).__init__(units, **kwargs)
        self.binary = binary
        self.H = H
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier

        super(BinaryDense, self).__init__(units, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[1]

        if self.H == 'Glorot':
            self.H = np.float32(np.sqrt(1.5 / (input_dim + self.units)))
        if self.kernel_lr_multiplier == 'Glorot':
            self.kernel_lr_multiplier = np.float32(1. / np.sqrt(1.5 / (input_dim + self.units)))

        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      #dtype =tf.float32,
                                      trainable = True)
        self.binary_kernel = self.add_weight(shape=(input_dim, self.units),
                                             initializer=self.kernel_initializer,
                                             name='binary',
                                             #dtype = tf.float32,
                                      trainable=False)

        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        constraint=self.bias_constraint)

        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None


        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
        #self.binary_kernel = binarize(self.kernel, H=self.H)
        self.binary_kernel =K.update(self.binary_kernel, binarize(self.kernel, H=self.H))
        #K.update(self.binary_kernel, K.round(self.kernel))
        tf.add_to_collection(self.name + '_binary', self.binary_kernel)  # layer-wise group
        tf.add_to_collection('binary', self.binary_kernel)  # global group
        tf.add_to_collection('con_weight', self.kernel)  # global group
        tf.add_to_collection(self.name + 'con_weight', self.kernel)  # global group

    def call(self, inputs):

        output = K.dot(inputs, self.binary_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {'H': self.H,
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  'bias_lr_multiplier': self.bias_lr_multiplier}
        base_config = super(BinaryDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#####################################

class AdamOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adam algorithm.
    See [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    """

    def __init__(self, weight_scale, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 use_locking=False, name="Adam"):
        super(AdamOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        # BNN weight scale factor
        self._weight_scale = weight_scale

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        # Variables to accumulate the powers of the beta parameters.
        # Created in _create_slots when we know the variables to optimize.
        self._beta1_power = None
        self._beta2_power = None

        # Created in SparseApply if needed.
        self._updated_lr = None

    def _get_beta_accumulators(self):
        return self._beta1_power, self._beta2_power

    def _non_slot_variables(self):
        return self._get_beta_accumulators()

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and context.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = variable_scope.variable(self._beta1,
                                                            name="beta1_power",
                                                            trainable=False)
                self._beta2_power = variable_scope.variable(self._beta2,
                                                            name="beta2_power",
                                                            trainable=False)
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # for BNN kernel
        # origin version clipping weight method is new_w = old_w + scale*(new_w - old_w)
        # and adam update function is new_w = old_w - lr_t * m_t / (sqrt(v_t) + epsilon)
        # so subtitute adam function into weight clipping
        # new_w = old_w - (scale * lr_t * m_t) / (sqrt(v_t) + epsilon)
        scale = self._weight_scale[ var.name ] / 4

        return training_ops.apply_adam(
            var, m, v,
            math_ops.cast(self._beta1_power, var.dtype.base_dtype),
            math_ops.cast(self._beta2_power, var.dtype.base_dtype),
            math_ops.cast(self._lr_t * scale, var.dtype.base_dtype),
            math_ops.cast(self._beta1_t, var.dtype.base_dtype),
            math_ops.cast(self._beta2_t, var.dtype.base_dtype),
            math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
            grad, use_locking=self._use_locking).op

    def _resource_apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        return training_ops.resource_apply_adam(
            var.handle, m.handle, v.handle,
            math_ops.cast(self._beta1_power, grad.dtype.base_dtype),
            math_ops.cast(self._beta2_power, grad.dtype.base_dtype),
            math_ops.cast(self._lr_t, grad.dtype.base_dtype),
            math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
            math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
            math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
            grad, use_locking=self._use_locking)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t,
                               use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)
        v_sqrt = math_ops.sqrt(v_t)
        var_update = state_ops.assign_sub(var,
                                          lr * m_t / (v_sqrt + epsilon_t),
                                          use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
          grad.values, var, grad.indices,
          lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
              x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
            [resource_variable_ops.resource_scatter_add(
                x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(
            grad, var, indices, self._resource_scatter_add)

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(
                    self._beta1_power * self._beta1_t,
                    use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(
                    self._beta2_power * self._beta2_t,
                    use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                      name=name_scope)

def compute_grads(model,opt,loss,sess):
    layers = model.layers
    grads = []
    update_weights=[]

    for layer in layers:
        params = tf.get_collection(layer.name + "_binary")
        print('params',params)
        if params:
            #with sess:
            #    p=tf.get_default_session().run(params[0])

            #print('hhparams[0]',params[0])
            #print('p',type(p),p)
            #print (type(p))
            #p=tf.Variable(p)
            #print (type(p))
            grad = opt.compute_gradients(loss, params)
            #grad = tf.gradients(loss, params)
            #print('grad',grad[0])
            grads.append(grad[0][0])
            #grads.append(grad[0][0])

            update_weights.extend(params)

    return zip(grads, update_weights)


model = Sequential()
targets = K.placeholder(name="y", shape=(None, 1))
print ('targets type',type(targets))
# dense1
model.add(BinaryDense(10, input_shape=(input_dim,), H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,
                      name='dense1'))

model.add(Activation(binary_tanh, name='act1'))

output_tensor= model.output
#output_tensor = K.cast(output_tensor,tf.float32)
#output_tensor = tf.convert_to_tensor(output_tensor)
#print ('output type',type(output_tensor))

loss = tf.reduce_mean(tf.square(tf.maximum(0.,1.-targets*output_tensor)))
sess = tf.Session()

for var in tf.trainable_variables():
    print (var.name)

other_var = [var for var in tf.trainable_variables() if not var.name.endswith('binary:0')]
global_step1 = tf.Variable(0, trainable=False)
global_step2 = tf.Variable(0, trainable=False)
sess.run(tf.global_variables_initializer())

opt1 = AdamOptimizer("Glorot",0.01)
opt2 = tf.train.AdamOptimizer(0.001)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):   # when training, the moving_mean and moving_variance in the BN need to be updated.
    train_kernel_op = opt1.apply_gradients(compute_grads(model,opt1,loss,sess),  global_step=global_step1)
    train_other_op  = opt2.minimize(loss, var_list=other_var, global_step=global_step2)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output_tensor, 1), tf.argmax(targets, 1)), tf.float32))
old_acc = 0.0

def train_epoch(X, y, sess):
    a = sess.run([accuracy,train_kernel_op,train_other_op],
        feed_dict={model.input: X,
                    targets: y,
                    })
    print (a)

train_epoch([[1,2]],[1],sess)
#get_accuracy = K.function(inputs=input_tensors, outputs=compute_grads(model,opt,loss))


