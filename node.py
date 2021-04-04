import os
import sys

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for d in physical_devices:
    try: 
        tf.config.experimental.set_memory_growth(d, True) 
    except: 
        pass 

from keras import models
from keras import layers
from keras import optimizers
import keras.applications
import keras.callbacks
from keras import backend as K
from keras.utils.np_utils import to_categorical
import sklearn.metrics
import random
import pickle
import numpy as np
import time
import gc



METHOD = sys.argv[1]         # All, Random, Adversarial, Min
RUN = sys.argv[2]           # RunIndex
DIVISOR = sys.argv[3]       # 2, 4, 8, 16


assert METHOD in ['All', 'IID', 'OOD', 'Max', 'ADV', 'COV']
assert DIVISOR in ['2', '4', '8', '16']

print(f'METHOD is {METHOD}, divisor is {DIVISOR}, seed is {RUN}.')

OUTPUT_DIR = f'results/data/node/{METHOD}_{DIVISOR}/'
OUTPUT_DIR_MODEL = f'results/model/node/{METHOD}_{DIVISOR}/'

if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
    except:
        print ('Race condition!', os.path.exists(OUTPUT_DIR))

if not os.path.exists(OUTPUT_DIR_MODEL):
    try:
        os.makedirs(OUTPUT_DIR_MODEL)
    except:
        print ('Race condition!', os.path.exists(OUTPUT_DIR_MODEL))

STATSFILE = OUTPUT_DIR + RUN + '.p'
MODELFILE = OUTPUT_DIR_MODEL + RUN + '.h5'

print ('Working in', OUTPUT_DIR)
print ('Storing', STATSFILE)
print ('Storing', MODELFILE)


max_node = 99
min_node = 20
node_num = max_node - min_node + 1

val_num = int(node_num * 0.2)
test_num = int(node_num * 0.2)
train_num = int(round((node_num - val_num - test_num)/DIVISOR))

all_set = list(range(min_node, max_node + 1))
random.shuffle(all_set)

train_set = all_set

test_set = train_set[:test_num]
train_set = train_set[test_num:]

val_set = train_set[:val_num]
train_set = train_set[val_num:]

if METHOD == 'IID':
    train_set = train_set[: train_num]

if METHOD == 'OOD':
    train_set = sorted(train_set[: train_num])

if METHOD == 'ADV':
    distance = [min([abs(train-test) for test in test_set]) for train in train_set]
    train_set = sorted(train_set, reverse=True, 
                          key=lambda x:distance[train_set.index(x)]
                         )[: train_num]

if METHOD == 'COV':
    train_set = sorted(train_set)
    selected = []
    selected.append(train_set.pop(0))
    selected.append(train_set.pop(-1))
    
    for run in range(train_num - 2):
        distance = [min([abs(train-selected) for selected in selected]) for train in train_set]
        train_set = sorted(train_set, reverse=True, 
                           key=lambda x:distance[train_set.index(x)]
                          )
        selected.append(train_set.pop(0))
    
    train_set = selected
    
    
train_counter = 0
val_counter = 0
test_counter = 0
image_num = 100
train_target = 6000
val_target = 2000
test_target = 2000


X_train = np.zeros((train_target, 100, 100, 3), dtype=np.float32)
y_train = np.zeros((train_target, 1), dtype=np.float32)

X_val = np.zeros((val_target, 100, 100, 3), dtype=np.float32)
y_val = np.zeros((val_target, 1), dtype=np.float32)

X_test = np.zeros((test_target, 100, 100, 3), dtype=np.float32)
y_test = np.zeros((test_target, 1), dtype=np.float32)



t0 = time.time()

all_counter = 0


cnt = 0
for num in train_set:
    for index in range(train_target//train_num):
        img = cv2.imread(f'{IMAGE_FLOER}/{num}/{num}_{index}.png')
        img = img.astype(np.float32)
        img = img / 255.

        X_train[cnt] = img
        y_train[cnt] = num

        cnt += 1

cnt = 0
for num in val_set:
    for index in range(val_target//train_num):
        img = cv2.imread(f'{IMAGE_FLOER}/{num}/{num}_{index}.png')
        img = img.astype(np.float32)
        img = img / 255.

        X_val[cnt] = img
        y_val[cnt] = num

        cnt += 1

cnt = 0
for num in test_set:
    for index in range(test_target//test_num):
        img = cv2.imread(f'{IMAGE_FLOER}/{num}/{num}_{index}.png')
        img = img.astype(np.float32)
        img = img / 255.

        X_test[cnt] = img
        y_test[cnt] = num

        cnt += 1
        

# NORMALIZE DATA IN-PLACE (BUT SEPERATELY)
X_min = X_train.min()
X_max = X_train.max()
y_min = y_train.min()
y_max = y_train.max()

# scale in place
X_train -= X_min
X_train /= (X_max - X_min)
y_train -= y_min
y_train /= (y_max - y_min)

X_val -= X_min
X_val /= (X_max - X_min)
y_val -= y_min
y_val /= (y_max - y_min)

X_test -= X_min
X_test /= (X_max - X_min)
y_test -= y_min
y_test /= (y_max - y_min)

# normalize to -.5 .. .5
X_train -= .5
X_val -= .5
X_test -= .5

print ('memory usage', (X_train.nbytes + X_val.nbytes + X_test.nbytes +
                       y_train.nbytes + y_val.nbytes + y_test.nbytes) / 1000000., 'MB')


# FEATURE GENERATION
feature_time = 0

X_train_3D = np.stack((X_train,)*3, -1)
del X_train
gc.collect()
X_val_3D = np.stack((X_val,)*3, -1)
del X_val
gc.collect()
X_test_3D = np.stack((X_test,)*3, -1)
del X_test
gc.collect()

print ('memory usage', (X_train_3D.nbytes +
                        X_val_3D.nbytes + X_test_3D.nbytes) / 1000000., 'MB')

print(X_train_3D.shape, y_train.shape)

feature_generator = keras.applications.VGG19(
    include_top=False, input_shape=(100, 100, 3))

t0 = time.time()

# THE MLP
MLP = models.Sequential()   
MLP.add(layers.Flatten(input_shape=feature_generator.output_shape[1:]))
MLP.add(layers.Dense(256, activation='relu', input_dim=(100, 100, 3)))
MLP.add(layers.Dropout(0.5))
MLP.add(layers.Dense(1, activation='linear'))  # REGRESSION

model = keras.Model(inputs=feature_generator.input,
                    outputs=MLP(feature_generator.output))

sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd,
                metrics=['mse', 'mae'])  # MSE for regression

print(model.summary())


# TRAINING
t0 = time.time()

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto'),
             keras.callbacks.ModelCheckpoint(MODELFILE, monitor='val_loss', verbose=1, save_best_only=True, mode='OOD')]

history = model.fit(X_train_3D,
                    y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val_3D, y_val),
                    callbacks=callbacks,
                    verbose=True)

fit_time = time.time()-t0

print ('Fitting done', time.time()-t0)


# PREDICTION
y_pred = model.predict(X_test_3D)


# denormalize y_pred and y_test
y_test = y_test * (y_max - y_min) + y_min
y_pred = y_test * (y_max - y_min) + y_min

# compute MAE
MAE = np.mean(np.abs(y_pred - y_test))


# STORE
#   (THE NETWORK IS ALREADY STORED BASED ON THE CALLBACK FROM ABOVE!)
stats = dict(history.history)

stats['train_set'] = train_set
stats['val_set'] = val_set
stats['test_set'] = test_set
stats['y_min'] = y_min
stats['y_max'] = y_max
stats['y_test'] = y_test
stats['y_pred'] = y_pred
stats['MAE'] = MAE
stats['time'] = fit_time


with open(STATSFILE, 'wb') as f:
    pickle.dump(stats, f)

print ('MAE', MAE)
print ('Written', STATSFILE)
print ('Written', MODELFILE)
