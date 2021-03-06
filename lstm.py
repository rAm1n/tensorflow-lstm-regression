import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn
from tensorflow.contrib import layers as tflayers

def x_sin(x):
	return x * np.sin(x)

def sin_cos(x):
	return pd.DataFrame(dict(a=np.sin(x), b=np.cos(x)), index=x)

def rnn_data(data, time_steps, labels=False):
	"""
	creates new data frame based on previous observation
	  * example:
		l = [1, 2, 3, 4, 5]
		time_steps = 2
		-> labels == False [[1, 2], [2, 3], [3, 4]]
		-> labels == True [2, 3, 4, 5]
	"""
	rnn_df = []
	for i in range(len(data) - time_steps):
		if labels:
			try:
				rnn_df.append(data.iloc[i + time_steps -1 ].as_matrix())
			except AttributeError:
				rnn_df.append(data.iloc[i + time_steps -1 ])
		else:
			data_ = data.iloc[i: i + time_steps].as_matrix()
			rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

	return np.array(rnn_df, dtype=np.float32)


def split_data(data, val_size=0.1, test_size=0.1):
	"""
	splits data to training, validation and testing parts
	"""
	ntest = int(round(len(data) * (1 - test_size)))
	nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

	df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

	return df_train, df_val, df_test


def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
	"""
	Given the number of `time_steps` and some data,
	prepares training, validation and test data for an lstm cell.
	"""
	df_train, df_val, df_test = split_data(data, val_size, test_size)
	return (rnn_data(df_train, time_steps, labels=labels),
			rnn_data(df_val, time_steps, labels=labels),
			rnn_data(df_test, time_steps, labels=labels))

def load_csvdata(rawdata, time_steps, seperate=False):
	data = rawdata
	if not isinstance(data, pd.DataFrame):
		data = pd.DataFrame(data)
	#print(data)
	train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
	train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
	return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)

def load_csvdata_xy(rawdata_X, rawdata_y, time_steps, val_size=0.1, test_size=0.1):
	dataX = rawdata_X
	if not isinstance(dataX, pd.DataFrame):
		dataX = pd.DataFrame(dataX)
	dataY = rawdata_y
	if not isinstance(dataY, pd.DataFrame):
		dataY = pd.DataFrame(dataY)

	#print(data)
	train_x, val_x, test_x = prepare_data(dataX, time_steps, labels=False, val_size=val_size, test_size=test_size)
	train_y, val_y, test_yg = prepare_data(dataY, time_steps, labels=True, val_size=val_size, test_size=test_size)
	return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)

def generate_data(data, time_steps, seperate=False):
	"""generates data with based on a function fct"""
	# data = fct(x)
	# if not isinstance(data, pd.DataFrame):
		# data = pd.DataFrame(data)
	train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
	train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
	return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


def lstm_model(num_units, rnn_layers, dense_layers=None, learning_rate=0.1, optimizer='Adagrad'):
	"""
	Creates a deep model based on:
		* stacked lstm cells
		* an optional dense layers
	:param num_units: the size of the cells.
	:param rnn_layers: list of int or dict
						 * list of int: the steps used to instantiate the `BasicLSTMCell` cell
						 * list of dict: [{steps: int, keep_prob: int}, ...]
	:param dense_layers: list of nodes for each layer
	:return: the model definition
	"""

	def lstm_cells(layers):
		if isinstance(layers[0], dict):
			return [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(layer['num_units'],
																			   state_is_tuple=True),
												  layer['keep_prob'])
					if layer.get('keep_prob') else tf.contrib.rnn.BasicLSTMCell(layer['num_units'],
																				state_is_tuple=True)
					for layer in layers]
		return [tf.contrib.rnn.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]

	def dnn_layers(input_layers, layers):
		if layers and isinstance(layers, dict):
# r0.11                     return tflayers.stack(input_layers, tflayers.fully_connected,
# r0.10                        return learn.ops.dnn(input_layers,
			return tflayers.stack(input_layers, tflayers.fully_connected,
								 layers['layers'],
								 activation=layers.get('activation'),
								 dropout=layers.get('dropout'))
		elif layers:

# r0.10            return learn.ops.dnn(input_layers, layers)
# r0.11            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
			return tflayers.stack(input_layers, tflayers.fully_connected, layers)
		else:
			return input_layers

	def _lstm_model(X, y):
		stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
		x_ = tf.unpack(X, axis=1, num=num_units)
		output, layers = tf.nn.rnn(stacked_lstm, x_, dtype=dtypes.float32)
		output = dnn_layers(output[-1], dense_layers)
		prediction, loss = learn.models.linear_regression(output, y)
		train_op = tf.contrib.layers.optimize_loss(
			loss, tf.contrib.framework.get_global_step(), optimizer=optimizer,
			learning_rate=learning_rate)
		return prediction, loss, train_op

	return _lstm_model
