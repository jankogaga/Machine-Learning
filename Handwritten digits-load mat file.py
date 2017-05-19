from keras.models import Sequential
from keras.layers import Dense
import numpy
import scipy.io as sio
import matplotlib.pyplot as plt
from random import randint

#initialisation
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10

# fix random seed for reproducibility
numpy.random.seed(7)
# load .matlab dataset from ex4 NN, each image 20x20px
dataset = sio.loadmat('ex4data1.mat')

# split into input (X) and output (Y) variables
X = dataset['X']
y = dataset['y']

#Display the first digit; need to reshape to 3 dimensional array (X1)
X1 = numpy.reshape(X,(len(y),20,20))
print("\nDisplay the first digit") 
plt.figure(1, figsize=(3, 3))
plt.imshow(X1[0,:,:], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

#For python we need to convert 10 to 0
for i in range(0, len(y)-1):
	if y[i] == 10: 
		y[i] = 0

#To code digits to vector e.g. for digit 0 to [1000000000]
Ident = numpy.identity (num_labels)
dim = (len(y),num_labels)
Y = numpy.zeros (dim) #5000x10
for i in range(0, len(y)):
	Y[i,:] = Ident[y[i],:]

# create model
model = Sequential()
model.add(Dense(hidden_layer_size, input_dim=input_layer_size, activation='relu'))
model.add(Dense(input_layer_size, activation='relu'))
model.add(Dense(num_labels, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #adam is Stochastic gradient descent
# Fit the model
model.fit(X, Y, epochs=10, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# calculate predictions
predictions = model.predict(X)
#Decoded matrix to array of digits
dim = (len(y),num_labels)
P = numpy.zeros (len(y)) #5000x1
for i in range(0, len(y)):
	P[i] = numpy.argmax(predictions[i])

#Randomly image prediction 3 times
for i in range(0,3):
	rand = randint(1, len(y))
	plt.figure(1, figsize=(2, 2))
	plt.imshow(X1[rand], cmap=plt.cm.gray_r, interpolation='nearest')
	plt.show()
	print (P[rand])


