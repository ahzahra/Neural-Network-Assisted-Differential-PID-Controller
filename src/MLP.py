import numpy as np
import numdifftools as nd
from sklearn.metrics import mean_squared_error


X = np.ones((10,3))
y = np.ones((10,3))
n_nodes = 4
layers = 2
epochs = 1

r_x,c_x = X.shape
r_y,c_y = y.shape
n = ((c_x+1)*n_nodes) + ((n_nodes+1)*n_nodes*(layers-1)) + (n_nodes+1)*c_y
bias = np.ones(layers + 1)
thetas = np.random.rand(n)
fprop = []
outputs = np.zeros(y.shape)

# forward prop
for i in xrange(r_x): # loop through pairs
    pair = X[i,:]
    ind_start = 0
    ind_end = c_x + 1
    ind_bias = 0
    # loop through transition into first hidden layer
    a = []
    for j in xrange(n_nodes):
        bias_pair = np.append(bias[ind_bias],pair)
        theta = thetas[ind_start:ind_end]
        a.append(np.dot(bias_pair,theta.T))
        ind_start = ind_end
        ind_end = ind_end + c_x + 1

    a = np.tanh(a)
    fprop.append(a)
    ind_end = ind_end + (n_nodes - c_x)

    # loop through rest of layers
    for j in xrange(layers-1):
        ind_bias = ind_bias + 1
        a_new = []
        for k in xrange(n_nodes):
            bias_a = np.append(bias[ind_bias],a)
            theta = thetas[ind_start:ind_end]
            a_new.append(np.dot(bias_a,theta.T))
            ind_start = ind_end
            ind_end = ind_end + n_nodes + 1
        a_new = np.tanh(a_new)
        fprop.append(a_new)
        a = a_new

    # loop through transition into output
    for j in xrange(c_y):
        bias_a = np.append(bias[ind_bias],a)
        theta = thetas[ind_start:ind_end]
        outputs[i,j] = np.dot(bias_a,theta.T)
        ind_start = ind_end
        ind_end = ind_end + n_nodes + 1

print outputs
# back prop/update