# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: example1_or_nn.py
@time: 2016/10/11 0:07

Use Hamaa to build a mlp to solve the "or problem".
This case includes all the necessary functions
for creating, training and testing a neural network.
"""

from hamaa.datasets import datasets
from hamaa.layers import Dense, Activation
from hamaa.models import Sequential
from hamaa.optimizers import SGD


# 1. create a model
model = Sequential()

# 2. add a full connected layer to model
model.add(Dense(input_dim=2, output_dim=2, init='uniform'))

# 3. add a activation layer to model
model.add(Activation('sigmoid'))

# 4. use "mean square error" as the objective of model
model.set_objective('mse')

# 5. use "stochastic gradient descent" as the optimizerof model
model.set_optimizer(SGD(lr=0.9, momentum=0.9, decay=1e-6))

# 6. print the summary of model
print model.summary()

# 7. load "or" data, note that the label "y" is one hot
#    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#    y = np.array([[1, 0], [0, 1], [0, 1], [0, 1]])
x, y = datasets.load_or_data()

# 8. train the neural network
model.train(training_data=(x, y), nb_epochs=10)

# 9. evaluate the accuracy on data
print 'test accuracy: ', model.evaluate_accuracy(x, y)

