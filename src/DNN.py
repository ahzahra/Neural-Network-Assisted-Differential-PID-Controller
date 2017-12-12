from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score

X_FEATURE = 'x'  # Name of the input feature.
N_CLASSES = 24
N_LAYERS = 5

def my_model(features, labels, mode, params):
  """DNN with three hidden layers, and dropout of 0.1 probability."""
  
  # Note how the variable "net" is repeatedly used as the input to the next layer, 
  # then updated.  This creates a net that looks like:
  #   features -> layer10 -> dropout -> layer20 -> dropout -> layer10 -> dropout -> argmax_logits
  # where "layer##" is a fully connected relu layer with ## units
  
  # Create three fully connected layers respectively of size 10, 20, and 10 with
  # each layer having a dropout probability of 0.1
  net = features[X_FEATURE]

  # Retrieve the layers and the activation function from the params
  layers = params.get('layers')
  activation_func = params.get('activation_func')
  dropout_rates = params.get('dropout')

  for units,rate in zip(layers,dropout_rates):
    net = tf.layers.dense(net, units=units, activation=activation_func)
    net = tf.layers.dropout(net, rate=rate)

  # Compute logits (1 per class)
  logits = tf.layers.dense(net, N_CLASSES, activation=None)

  # Compute predictions via the argmax over the logits
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode, predictions=logits)

  loss = tf.losses.mean_squared_error(labels=labels, predictions = logits)

  # Create training optimizer, using adagrad to minimize the cross entropy loss
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  # Compute evaluation metrics
  eval_metric_ops = {
      'error': tf.metrics.mean_squared_error(
          labels=labels, predictions=logits)
  }
  
  return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

def train_and_evaluate(layers, dropout_rate, x_train, x_test, y_train, y_test, activation_func):
  params = {'layers': layers, 'activation_func': activation_func, 'dropout': dropout_rate}
  classifier = tf.estimator.Estimator(model_fn=my_model, params = params)

  # train the model for 1000 steps
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={X_FEATURE: x_train}, y=y_train, num_epochs=None, shuffle=True)
  classifier.train(input_fn=train_input_fn, steps=1000)

  # predict classes for the test data
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={X_FEATURE: x_test}, y=y_test, num_epochs=1, shuffle=False)
  predictions = classifier.predict(input_fn=test_input_fn)

  y_predicted = np.zeros((y_test.shape))
  for i,p in enumerate(predictions):
    y_predicted[i,:] = p

  # compute the mean squared error via sklearn's built-in function
  score = metrics.mean_squared_error(y_test, y_predicted)

  print(y_test[0,:], y_predicted[0,:])
  return score

def main(unused_argv):

  # load the features
  filename = 'data.txt'
  data = np.loadtxt(filename, delimiter=' ')
  X = data[:,0:5]
  Y1 = data[:,5:17]
  Y2 = data[:,17:]
  Y = data[:,5:]

  # Standardize the features 
  mu_X = X.mean(axis = 0)
  sigma_X = X.std(axis = 0)
  X_standardized = (X - mu_X)/sigma_X

  # Standardize the labels
  mu_Y = Y.mean(axis = 0)
  sigma_Y = Y.std(axis = 0)
  Y_standardized = (Y - mu_Y)/sigma_Y

  layers = 100*np.ones((N_LAYERS,1))
  dropout_rate1 = 0.1*np.ones((5,1))

  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      X_standardized, Y_standardized, test_size=0.2, random_state=42)

  scores_relu = train_and_evaluate(layers, dropout_rate1, x_train, x_test, y_train, y_test, tf.nn.relu)

  print(scores_relu)

  # # scores_relu = np.zeros((10,1))
  # # scores_tanh = np.zeros((10,1))
  # # scores_logistic = np.zeros((10,1))

  # scores_dr1 = np.zeros((10,1))
  # scores_dr2 = np.zeros((10,1))
  # scores_dr3 = np.zeros((10,1))

  # for i in range(10):
  #   # Set the number of layers and the number of nodes in each layer
  #   layers = 100*np.ones((1*(i+1),1))
  #   dropout_rate1 = 0.1*np.ones((1*(i+1),1))
  #   dropout_rate2 = np.linspace(0.1,0.7, 5*(i+1))
  #   dropout_rate3 = np.random.rand(5*(i+1),1)


  #   # scores_relu[i] = train_and_evaluate(layers, dropout_rate1, x_train, x_test, y_train, y_test, tf.nn.relu)
  #   # scores_tanh[i] = train_and_evaluate(layers, dropout_rate1, x_train, x_test, y_train, y_test, tf.nn.tanh)
  #   # scores_logistic[i] = train_and_evaluate(layers, dropout_rate1, x_train, x_test, y_train, y_test, tf.nn.sigmoid)

  #   scores_dr1[i] = train_and_evaluate(layers, dropout_rate1, x_train, x_test, y_train, y_test, tf.nn.tanh)
  #   scores_dr2[i] = train_and_evaluate(layers, dropout_rate2, x_train, x_test, y_train, y_test, tf.nn.tanh)
  #   scores_dr3[i] = train_and_evaluate(layers, dropout_rate3, x_train, x_test, y_train, y_test, tf.nn.tanh)

  # no_layers = np.linspace(1, 10, 10)
  # # plt.plot(no_layers, scores_relu, label = 'Relu Activation 0.1 dropout')
  # # plt.plot(no_layers, scores_tanh, label = 'Tanh Activation 0.1 dropout')
  # # plt.plot(no_layers, scores_logistic, label = 'Sigmoid Activation 0.1 dropout')

  # plt.plot(no_layers, scores_dr1, label = 'Tanh Activation with dropout 0.1')
  # plt.plot(no_layers, scores_dr2, label = 'Tanh Activation with increasing dropout_rate')
  # plt.plot(no_layers, scores_dr3, label = 'Tanh Activation with random dropout_rate')
  
  # plt.legend()
  # plt.xlabel('Number of hidden layers')
  # plt.ylabel('Training Set Score')
  # plt.title('Performance for different dropouts (100 Nodes per layer)')
  # plt.grid(True)
  # # plt.savefig('graphTextClassifierROC.pdf')
  # plt.show()


if __name__ == '__main__':
  tf.app.run()