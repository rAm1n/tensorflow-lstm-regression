import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error
from lstm import generate_data, lstm_model, load_csvdata, load_csvdata_xy, sin_cos
import dateutil.parser
import datetime
import matplotlib.dates as mdates
import os
import shutil
import math
import random

# When you change network parameters, Do rm -rf ops_logs/*
LOG_DIR = './ops_logs/dnn_non_trendy'
TIMESTEPS = 10
#RNN_LAYERS = [{'num_units': 10}] #, {'num_units': 2}]
#DENSE_LAYERS = None #[ 2, 2 ] #None #[30]
TRAINING_STEPS = 3200000 #100000 #3200000
BATCH_SIZE = 100
PRINT_STEPS = 365
NORMALFACTOR = 1000.0

try:
	shutil.rmtree(LOG_DIR)
except OSError:
	pass

def generate_non_trendy_data( max, nstep ):
    #t = np.linspace(0, max, nstep, dtype=np.float32)
#    x1 = [random.uniform(0.0, 1.0) for i in xrange(nstep)]
#    x2 = [random.uniform(0.0, 1.0) for i in xrange(nstep)]
    x1 = np.linspace(0, max, nstep, dtype=np.float32)
    x2 = np.linspace(0, max, nstep, dtype=np.float32)
    y = x1
    X = np.column_stack((x1,x2))
    return pd.DataFrame(X), pd.DataFrame(y)

def spread_inputs(a):
    col = np.shape(a)[1] * np.shape(a)[2]
    return np.reshape(a, (-1, col))

data_X, data_y = generate_non_trendy_data(40*math.pi, 5000)  # max, num_steps

X, y = load_csvdata_xy(data_X, data_y, TIMESTEPS, val_size=0.01, test_size=0.10)
X['train'] = spread_inputs(X['train'])
X['val'] = spread_inputs(X['val'])
X['test'] = spread_inputs(X['test'])

inputsize = np.shape(X['train'])[1]

print X['train'][0], y['train'][0]

estimator = learn.DNNRegressor(feature_columns=None, hidden_units = [inputsize*2, inputsize*2], model_dir=LOG_DIR)

estimator.fit(x=X['train'], y=y['train'], steps=TRAINING_STEPS, batch_size=BATCH_SIZE)


#net = learn.input_data(shape=[None, inputsize])
#net = learn.fully_connected(net, inputsize*2)
#net = learn.fully_connected(net, inputsize*2)
#net = learn.fully_connected(net, 1, activation='ReLU')
#net = learn.regression(net)

#model = learn.DNN(net)
#model.fit(X['train'], y['train'], n_epoch=TRAINING_STEPS, batch_size=BATCH_SIZE, show_metric=True)


predicted = estimator.predict(x=X['test'])
#not used in this example but used for seeing deviations
#rmse = np.sqrt(((predicted - y['test']) ** 2).mean(axis=0))
score = mean_squared_error(predicted, y['test'])
print ("MSE: %f" % score)

plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(y['test'], label='test')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()

