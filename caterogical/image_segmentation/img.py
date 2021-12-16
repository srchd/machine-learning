import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.models
import tensorflow.keras.optimizers
import tensorflow.keras.layers
import tensorflow.keras.activations
import tensorflow.keras.callbacks
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import os

np.random.seed()

data = pd.read_csv('datasets/segmentation.test', sep=',')

np_data = data.to_numpy(dtype=np.float32)
X = np_data
Y = np.zeros(X.shape[0])

print(X.shape)
print(Y.shape)

CAT_NAMES = ['BRICKFACE', 'SKY', 'FOLIAGE', 'CEMENT', 'WINDOW', 'PATH', 'GRASS']

for idx, cat in enumerate(data.index):
    Y[idx] = CAT_NAMES.index(cat)

Y = to_categorical(Y)

# Mixing input and output arrays

perm = np.random.permutation(Y.shape[0])

X = X[perm]
Y = Y[perm]

# Making train (50%), val (25%) and test(25%)

x_train, x_val, x_test = np.split(X, [int(.5 * len(X)), int(.75 * len(X))])
y_train, y_val, y_test = np.split(Y, [int(.5 * len(Y)), int(.75 * len(Y))])

opt = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9)

reg_model = tf.keras.models.Sequential()
reg_model.add(tf.keras.layers.Dense(50, activation="relu", input_dim=x_train.shape[1], kernel_initializer='he_uniform'))
reg_model.add(tf.keras.layers.Dense(30, activation='relu'))
reg_model.add(tf.keras.layers.Dense(20, activation='relu'))
reg_model.add(tf.keras.layers.Dense(7, activation='softmax'))
reg_model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

model_checkpoint_folder = './chkpts'
LOAD_MODEL_FROM_CHECKPOINT = True
checkpoint_name_to_load = 'cp-0245.ckpt'

if LOAD_MODEL_FROM_CHECKPOINT and os.path.isdir(model_checkpoint_folder) and os.path.isfile(os.path.join(model_checkpoint_folder, checkpoint_name_to_load + '.index')):
    checkpoint_path = os.path.join(model_checkpoint_folder, checkpoint_name_to_load)
    reg_model.load_weights(checkpoint_path)
    print('Model loaded from checkpoint:', checkpoint_path)
    
else:
    print('Checkpoint is disabled, or not found. Training model...')
    
    model_chkpt = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_checkpoint_folder, \
                                                        'cp-{epoch:04d}.ckpt'), verbose=1, \
                                                        save_weights_only=True, save_freq=400)
                                                        

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=30, \
                                                        restore_best_weights=True)

    history = reg_model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=60, epochs=300, verbose=1, callbacks=[early_stopping, model_chkpt])

    # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    #accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.savefig('acc-loss.png')
    plt.show()
    
_, train_acc = reg_model.evaluate(x_train, y_train, verbose=0)
_, test_acc = reg_model.evaluate(x_test, y_test, verbose=0)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# predict thingies

pred_data = pd.read_csv('datasets/segmentation.data', sep=',')
x_pred = pred_data.to_numpy(dtype=np.float32)
y_pred = np.zeros(x_pred.shape[0])

for idx, cat in enumerate(pred_data.index):
    y_pred[idx] = CAT_NAMES.index(cat)

    
sample_idxs = np.random.choice(x_pred.shape[0], size=10)
predictions = reg_model.predict(x_pred, verbose=0)

preds = np.argmax(predictions, axis=1)

for i in range(10):
    print(f'{sample_idxs[i]}. data => prediction: {CAT_NAMES[preds[sample_idxs[i]]]} (real value: {CAT_NAMES[y_pred[sample_idxs[i]].astype(int)]})')