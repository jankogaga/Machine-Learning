from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn import datasets
import matplotlib.pyplot as plt
from random import randint

#initialisation
input_layer_size  = 64  # 8x8 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10

# fix random seed for reproducibility
numpy.random.seed(7)
#Load the digits dataset
dataset = datasets.load_digits()

#Display the first digit
plt.figure(1, figsize=(3, 3))
plt.imshow(dataset.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# split into input (X) and output (Y) variables
X = dataset['images']	#1797x8x8
Y = dataset['target']	#1797x1
X = X.reshape(len(Y),input_layer_size)	#1797x64

#To code digits to vector e.g. for digit 0 to [1000000000] result is YC
Ident = numpy.identity (num_labels)
dim = (len(Y),num_labels)
YC = numpy.zeros (dim) #1797x10
for i in range(0, len(Y)):
	YC[i,:] = Ident[Y[i],:]

Y = YC
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
dim = (len(Y),num_labels)
P = numpy.zeros (len(Y)) 
for i in range(0, len(Y)):
	P[i] = numpy.argmax(predictions[i])

#Randomly image prediction 3 times
for i in range(0,3):
	rand = randint(1, len(Y))
	plt.figure(1, figsize=(2, 2))
	plt.imshow(dataset.images[rand], cmap=plt.cm.gray_r, interpolation='nearest')
	plt.show()
	print (P[rand])




