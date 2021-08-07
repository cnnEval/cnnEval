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

import ClevelandMcGill as C


EXPERIMENT = sys.argv[1]    # e.g. type1
METHOD = sys.argv[2]         # All, Random, Adversarial, Min
RUN = sys.argv[3]          # RunIndex
DIVISOR = sys.argv[4]       # 2, 4, 8, 16


assert EXPERIMENT in [f'type{i}' for i in range(1, 6)]
assert METHOD in ['All', 'IID', 'OOD', 'Max', 'ADV', 'COV']
assert DIVISOR in ['2', '4', '8', '16']

print(f'Running {EXPERIMENT}, split is {METHOD}, divisor is {DIVISOR}, seed is {RUN}.')

DATATYPE = eval(f'C.Figure4.data_to_{EXPERIMENT}')

OUTPUT_DIR = f'results/data/unconstrained_ratio/{METHOD}_{DIVISOR}/{EXPERIMENT}/'
OUTPUT_DIR_MODEL = f'results/model/unconstrained_ratio/{METHOD}_{DIVISOR}/{EXPERIMENT}/'

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



# RATIO GENERATION
if EXPERIMENT != 'type5':
    H = [float(i) for i in range(5, 86)]
else:
    H = [float(i) for i in range(5, 46)]

P = [[i, j] for i in H for j in H if i < j]

ratio_pair = {} # {r1:[p1,p2...], r2:[p1, p2...]}
for p in P:
    r = round(min(p)/max(p), 2)
    if r not in ratio_pair:
        ratio_pair[r] = [p]
    else:
        ratio_pair[r].append(p)

all_ratios = list(ratio_pair.keys())
random.shuffle(all_ratios)

testNum = int(round(len(all_ratios) * 0.2))
valNum = int(round(len(all_ratios) * 0.2))
trainNum = len(all_ratios) - valNum - testNum
# test ratios:
test_ratios = all_ratios[:testNum]
# val ratios:
val_ratios = all_ratios[testNum:testNum+valNum]
# train_ratios, all by default
train_ratios = all_ratios[testNum+valNum:]

if METHOD == 'IID':
    train_ratios = train_ratios[:int(round(trainNum // int(DIVISOR)))]
if METHOD == 'OOD':
    train_ratios = sorted(train_ratios)[:int(round(trainNum // int(DIVISOR)))]
if METHOD == 'ADV':
    distance = [min([abs(train-test) for test in test_ratios]) for train in train_ratios]
    train_ratios = sorted(train_ratios, reverse=True, 
                            key=lambda x:distance[train_ratios.index(x)]
                            )[:int(round(trainNum // int(DIVISOR)))]
if METHOD == 'COV':
    train_ratios = sorted(train_ratios)
    selected_ratios = []
    selected_ratios.append(train_ratios.pop(0))
    selected_ratios.append(train_ratios.pop(-1))
    
    for run in range(int(round(trainNum // int(DIVISOR))) - 2):
        distance = [min([abs(train-selected) for selected in selected_ratios]) for train in train_ratios]
        train_ratios = sorted(train_ratios, reverse=True, 
                                key=lambda x:distance[train_ratios.index(x)]
                                )
        selected_ratios.append(train_ratios.pop(0))
    
    train_ratios = selected_ratios




# DATA GENERATION
train_counter = 0
val_counter = 0
test_counter = 0
train_target = 60000
val_target = 20000
test_target = 20000

X_train = np.zeros((train_target, 100, 100), dtype=np.float32)
y_train = np.zeros((train_target, 1), dtype=np.float32)

X_val = np.zeros((val_target, 100, 100), dtype=np.float32)
y_val = np.zeros((val_target, 1), dtype=np.float32)

X_test = np.zeros((test_target, 100, 100), dtype=np.float32)
y_test = np.zeros((test_target, 1), dtype=np.float32)

y_data = np.zeros((test_target, 2), dtype=np.float32)


t0 = time.time()
all_counter = 0

while train_counter < train_target:
    all_counter += 1
    ratio = random.choice(train_ratios)
    data = random.choice(ratio_pair[ratio])
    random.shuffle(data)
    label = min(data) / max(data)
    
    try:
        image = DATATYPE(data)
        image = image.astype(np.float32)
    except:
        continue
    
    image += np.random.uniform(0, 0.05, (100, 100))
    
    X_train[train_counter] = image
    y_train[train_counter] = label
    train_counter += 1


while val_counter < val_target:
    all_counter += 1
    
    ratio = random.choice(val_ratios)
    data = random.choice(ratio_pair[ratio])
    random.shuffle(data)
    label = min(data) / max(data)

    try:
        image = DATATYPE(data)
        image = image.astype(np.float32)
    except:
        continue
    
    image += np.random.uniform(0, 0.05, (100, 100))
    
    X_val[val_counter] = image
    y_val[val_counter] = label
    val_counter += 1


while test_counter < test_target:
    all_counter += 1
    ratio = random.choice(test_ratios)
    data = random.choice(ratio_pair[ratio])
    random.shuffle(data)
    label = min(data) / max(data)
    
    try:
        image = DATATYPE(data)
        image = image.astype(np.float32)
    except:
        continue
    
    image += np.random.uniform(0, 0.05, (100, 100))
    
    X_test[test_counter] = image
    y_test[test_counter] = label
    y_data[test_counter] = data
    test_counter += 1

print ('Done', time.time()-t0, 'seconds (', all_counter, 'iterations)')


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
y_pred = y_pred * (y_max - y_min) + y_min

# compute MAE
MAE = np.mean(np.abs(y_pred - y_test))


# STORE
#   (THE NETWORK IS ALREADY STORED BASED ON THE CALLBACK FROM ABOVE!)
stats = dict(history.history)

stats['train_ratios'] = train_ratios
stats['val_ratios'] = val_ratios
stats['test_ratios'] = test_ratios
stats['y_min'] = y_min
stats['y_max'] = y_max
stats['y_data'] = y_data
stats['y_test'] = y_test
stats['y_pred'] = y_pred
stats['MAE'] = MAE
stats['time'] = fit_time

with open(STATSFILE, 'wb') as f:
    pickle.dump(stats, f)

print ('MAE', MAE)
print ('Written', STATSFILE)
print ('Written', MODELFILE)
