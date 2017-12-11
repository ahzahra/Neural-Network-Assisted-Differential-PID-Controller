import numpy as np
import numdifftools as nd
from sklearn.metrics import mean_squared_error
from scipy import optimize

filename = "data.txt"
data = np.loadtxt(filename, delimiter=' ')
X = data[:,:5]
y = data[:,5:]

n_nodes = 4
layers = 1
epochs = 1

eps = 1e-6
# eps = 0

r_x,c_x = X.shape
r_y,c_y = y.shape
N = ((c_x+1)*n_nodes) + ((n_nodes+1)*n_nodes*(layers-1)) + (n_nodes+1)*c_y
bias = np.ones(layers + 1)
thetas = np.random.rand(N)
outputs = np.zeros(y.shape)
nP = outputs.shape[0]*outputs.shape[1]
rho = 1
rho = rho * np.eye(N)


for ep in xrange(epochs):
    print "===================EPOCH %d===================" %(ep)
    orig_err = []
    j_thetas = np.copy(thetas)
    jacobian = np.zeros((nP,N))
    # calculate jacobian error values
    for j_ind in xrange(N+1):
        print j_ind
        # increment a different theta for each iteration
        if j_ind != 0:
            j_thetas[j_ind-1] += eps
            # undo previous eps increment
            if j_ind != 1:
                j_thetas[j_ind-1-1] -= eps

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
                theta = j_thetas[ind_start:ind_end]
                a.append(np.dot(bias_pair,theta.T))
                ind_start = ind_end
                ind_end = ind_end + c_x + 1

            a = np.tanh(a)
            ind_end = ind_end + (n_nodes - c_x)

            # loop through rest of layers
            for j in xrange(layers-1):
                ind_bias = ind_bias + 1
                a_new = []
                for k in xrange(n_nodes):
                    bias_a = np.append(bias[ind_bias],a)
                    theta = j_thetas[ind_start:ind_end]
                    a_new.append(np.dot(bias_a,theta.T))
                    ind_start = ind_end
                    ind_end = ind_end + n_nodes + 1
                a_new = np.tanh(a_new)
                a = a_new

            # loop through transition into output
            for j in xrange(c_y):
                bias_a = np.append(bias[ind_bias],a)
                theta = j_thetas[ind_start:ind_end]
                outputs[i,j] = np.dot(bias_a,theta.T)
                ind_start = ind_end
                ind_end = ind_end + n_nodes + 1

        # calculate error matrix
        err = y - outputs
        err = err.reshape((err.shape[0]*err.shape[1],1))
        if j_ind == 0:
            orig_err = np.copy(err)
        else:
            jacobian[:,j_ind-1] = err.reshape((err.shape[0],))
    # compute jacobian matrix
    jacobian = (jacobian - orig_err) / eps
    ainv = np.linalg.inv(rho + np.dot(jacobian.T,jacobian))
    thetas = thetas - (np.dot(ainv, np.dot(jacobian.T,orig_err))).reshape((thetas.shape[0],))
