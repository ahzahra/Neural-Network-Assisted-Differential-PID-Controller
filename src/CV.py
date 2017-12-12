import numpy as np
import numdifftools as nd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from scipy import optimize
import MLP

filename = "data.txt"
data = np.loadtxt(filename, delimiter=' ')
Xdata = data[:,:5]
ydata = data[:,5:]
eps = 1e-12
K = 15
num_of_layers = 1
num_of_neurones = [100]

kf = KFold(n_splits=K, shuffle=True)

net = MLP.init(eps, num_of_layers, num_of_neurones)

for train_index, test_index in kf.split(Xdata):
    net.train(Xdata[train_index])
    y_predict = net.predict(Xdata[test_index])
    print(accuracy_score(ydata[test_index], y_predict))