import numpy as np
import tensorflow as tf
import tensorflow.keras.models
import tensorflow.keras.optimizers
import tensorflow.keras.layers
import tensorflow.keras.activations
import tensorflow.keras.callbacks

import matplotlib.pyplot as plt

np.random.seed()

with open('dataset/student-mat_prep.csv', 'r') as file:
    content = file.read()

# content = np.loadtxt('dataset/student-mat.csv')

# print(len(content))
# print(content[:500])

lines = content.split('\n')
words = [line.split(';') for line in lines]
vals = words[1:len(words) - 1]
vals = [[float(item) for item in rec[:-1]] for rec in vals]
features = np.array(vals, dtype=np.float32)
labs = words[1:len(words) - 1]
labs = [float(item[-1]) for item in labs]
labels = np.array(labs, dtype=np.float32)

# train: 50% / val: 25% / test: 25%

perm = np.random.permutation(labels.shape[0])

features = features[perm]
labels = labels[perm]

f_len = len(features)
l_len = len(labels)

x_unnorm_train, x_unnorm_val, x_unnorm_test = np.split(features, [int(.5 * f_len), int(.75 * f_len)])
y_train, y_val, y_test = np.split(labels, [int(.5 * l_len), int(.75 * l_len)])

print(x_unnorm_train.shape)
print(x_unnorm_val.shape)
print(x_unnorm_test.shape)

print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

def normalize_min_max(arr, axis):
      return arr - arr.min(axis=axis)

def denorm_min_max(arr, axis):
  return arr.max(axis=axis) - arr.min(axis=axis)

x_train = normalize_min_max(x_unnorm_train, 0) / denorm_min_max(x_unnorm_train, 0)
x_val = (x_unnorm_val - x_unnorm_train.min(axis=0)) / denorm_min_max(x_unnorm_train, 0)
x_test = (x_unnorm_test - x_unnorm_train.min(axis=0)) / denorm_min_max(x_unnorm_train, 0)

reg_model = tf.keras.models.Sequential()
reg_model.add(tf.keras.layers.Dense(50, activation="relu", input_dim=x_train.shape[1]))
reg_model.add(tf.keras.layers.Dropout(0.3))
reg_model.add(tf.keras.layers.Dense(1))
reg_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005), loss="mse", metrics=["mae"])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, )
history = reg_model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=200, verbose=1, callbacks=[early_stopping])

tr_mse = history.history['loss']
val_mse = history.history['val_loss']

tr_mae = history.history['mae']
val_mae = history.history['val_mae']

plt.figure(figsize=(7,5))
plt.subplot(2, 1, 2)

plt.plot(tr_mae, label="tr_mae")
plt.plot(val_mae, label="val_mae")
plt.ylim((0,30))
plt.xlabel("Number of epochs")
plt.ylabel("Cost (J)")
plt.legend()
plt.show()

test_mse, test_mae = reg_model.evaluate(x_test, y_test)