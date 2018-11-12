from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
import numpy as np
from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense
from data import X_train,y_train,X_test,y_test


def binary_tanh(x):
    return binary_tanh_op(x)


H = 1.
w_lr_multiplier = 'Glorot'

# nn
batch_size = 96
epochs = 50
input_dim=40
classes = 1
use_bias = True

# learning rate schedule
lr_start = 1e-4
lr_end = 1e-5
lr_decay = (lr_end / lr_start)**(1. / epochs)



# BN
epsilon = 1e-6
momentum = 0.9


#model
model = Sequential()

# dense1
model.add(Dense(30,input_shape=(input_dim,),use_bias=True,activation='sigmoid'))
#model.add(BinaryDense(10, input_shape=(input_dim,),H=H, w_lr_multiplier=w_lr_multiplier, use_bias=use_bias, name='dense1'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn1'))
model.add(Activation(binary_tanh, name='act1'))


# dense2
model.add(BinaryDense(classes, input_shape=(10,),H=H, w_lr_multiplier=w_lr_multiplier, use_bias=use_bias, name='dense2',activation='sigmoid'))

#model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn2'))
#model.add(Activation(binary_tanh, name='act2'))

opt = Adam(lr=lr_start)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
# loss = 'binary_crossentropy'


model.summary()

lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
history = model.fit(X_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(X_test, y_test),
                    callbacks=[lr_scheduler])
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


weights = np.array(model.get_weights())
print(weights.shape)
print(weights[0].shape)
print(weights[1].shape)

names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()

for name, weight in zip(names, weights):
    print(name, weight.shape)
#print(weights)
model.save_weights("/home/jiangnan/My_Binary_dAFM/Other/weight")
np.savetxt("/home/jiangnan/My_Binary_dAFM/Other/weight.txt",weights[0])

