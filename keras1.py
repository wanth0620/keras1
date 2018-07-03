import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense  #for fully connected layer
import matplotlib.pyplot as plt #for visualization

# create fake data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))

# plot data

plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]     #training data
X_test, Y_test = X[160:], Y[160:]       #testing data

model = Sequential()
model.add(Dence(output_dim = 1,input_dim = 1))
model.add(Dence(output_dim))





