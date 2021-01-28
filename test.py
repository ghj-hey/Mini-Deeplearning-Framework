import ghj_pkg.nn.core as tf
import ghj_pkg.utlis.utilities as util

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample

# from tqdm import tqdm_notebook
#from miniflow import *

# Load data
data = load_boston()

#正常15D数据输入
X_ = data['data']
y_ = data['target']

#2D数据输入
# dataframe = pd.DataFrame(data['data'])
# dataframe.columns = data['feature_names']
# X_ = dataframe[['RM', 'LSTAT']]
# y_ = data['target']

# Normalize data
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
n_hidden = 10
W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)

# Neural network
X, y = tf.Placeholder(), tf.Placeholder()
W1, b1 = tf.Placeholder(), tf.Placeholder()
W2, b2 = tf.Placeholder(), tf.Placeholder()

l1 = tf.Linear(X, W1, b1)
s1 = tf.Sigmoid(l1)
l2 = tf.Linear(s1, W2, b2)
cost = tf.MSE(y, l2)

feed_dict = {
    X: X_,
    y: y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}

epochs = 5000
# Total number of examples
m = X_.shape[0]
batch_size = 16
steps_per_epoch = m // batch_size

graph = util.topological_sort_feed_dict(feed_dict)
trainables = [W1, b1, W2, b2]

print("Total number of examples = {}".format(m))

losses = []

for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # Step 1
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        # Reset value of X and y Inputs
        X.value = X_batch
        y.value = y_batch

        # Step 2
        _ = None
        util.forward_and_backward(graph)  # set output node not important.

        # Step 3
        rate = 1e-2

        util.optimize(trainables, rate)

        loss += graph[-1].value

    if i % 100 == 0:
        print("Epoch: {}, Loss: {:.3f}".format(i + 1, loss / steps_per_epoch))
        losses.append(loss / steps_per_epoch)


