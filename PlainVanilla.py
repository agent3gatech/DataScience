import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# This can be used to reproduce results exactly after training. 
#tf.random.set_seed(0)

# Load data
a = np.load('mnist.npz')
x_train, l_train = a['x_train'], a['y_train']
x_test, l_test = a['x_test'], a['y_test']

# Transform the train label data into one-hot format
# This step is different from the tensorflow tutorial
# By using one-hot format, we can use a lot more loss functions.
y_train = np.zeros((l_train.shape[0], l_train.max()+1), dtype=np.float32)
y_train[np.arange(l_train.shape[0]), l_train] = 1
# Ditto for the test set
y_test = np.zeros((l_test.shape[0], l_test.max()+1), dtype=np.float32)
y_test[np.arange(l_test.shape[0]), l_test] = 1

# Renormalize the pixel info
x_train, x_test = x_train / 255.0, x_test / 255.0

# Plot a random image from the training dataset
#img = np.random.randint(1000)
#plt.imshow(x_train[img],cmap=plt.cm.binary)
#print("This entry has been labeled as ",l_train[img])
#plt.show()

# Create the network:
# Sequential NN with 3 layers
#    Input 784 (28,28)
#    Hidden 25
#    Output 10
# The activation function is a sigmoid - but only used on the hidden layer
# The output layers is simply the weighted sum of the hidden layer
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(25, activation='sigmoid'),
  tf.keras.layers.Dense(10)
])

# A standard Mean Squared Error loss function
loss_fn = tf.keras.losses.MeanSquaredError()

# SDG: steepest gradient descent
# report the accurady of the model during training
model.compile(optimizer='SGD',
              loss=loss_fn,
              metrics=['accuracy'])

# Do the training
# There are 5 training 'epochs'
# Use default mini_batch size
model.fit(x_train, y_train, epochs=5)

# Print statistics for evaluating the fit on the test data
model.evaluate(x_test,  y_test, verbose=2)

# The list predictions contains the output of the NN
# applied to the test data
# I print and plot the first 10 images
predictions = model.predict(x_test)
#for img in range(10):
#    hottest = np.argmax(predictions[img])
#    print(l_test[img],hottest)
#    plt.imshow(x_test[img],cmap=plt.cm.binary)
#    plt.show()
              
