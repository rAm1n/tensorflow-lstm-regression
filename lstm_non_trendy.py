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
LOG_DIR = './ops_logs/lstm_non_trendy'
TIMESTEPS = 10
RNN_LAYERS = [{'num_units': 10}] #, {'num_units': 2}]
DENSE_LAYERS = None #[ 2, 2 ] #None #[30]
TRAINING_STEPS = 100000 #3200000
BATCH_SIZE = 100
PRINT_STEPS = 365
NORMALFACTOR = 1000.0

try:
	shutil.rmtree(LOG_DIR)
except OSError:
	pass

def generate_non_trendy_data( max, nstep ):
    #t = np.linspace(0, max, nstep, dtype=np.float32)
    x1 = [random.uniform(0.0, 1.0) for i in xrange(nstep)]
    y =  x1
    X = x1
    return pd.DataFrame(X), pd.DataFrame(y)

data_X, data_y = generate_non_trendy_data(40*math.pi, 5000)  # max, num_steps
X, y = load_csvdata_xy(data_X, data_y, TIMESTEPS, val_size=0.01, test_size=0.10)

regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),
                           model_dir=LOG_DIR)

# create a lstm instance and validation monitor
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                     every_n_steps=PRINT_STEPS,
                                                     early_stopping_rounds=10000)
regressor.fit(X['train'], y['train'],
              monitors=[validation_monitor],
              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS)


predicted = regressor.predict(X['test'])
#not used in this example but used for seeing deviations
#rmse = np.sqrt(((predicted - y['test']) ** 2).mean(axis=0))
score = mean_squared_error(predicted, y['test'])
print ("MSE: %f" % score)

plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(y['test'], label='test')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()

