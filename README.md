# Machine-Learning
This is a portfolio for Machine Learning tasks written in Python.

Handwritten digits.py

represents Neural Network using Keras Python library.
Dataset used is built-in digit dataset. This dataset is made up of 1797 8x8 images. Each image is a hand-written digit.
For the training the network, we will use logarithmic loss, which for a binary classification problem is defined in Keras as “binary_crossentropy“. We will also use the efficient gradient descent algorithm “adam”. 
For just 10 iterations, accuracy of the network reaches 99.77%
In the end, the neural network chooses 3 random examples from dataset and makes a prediction for them.

Handwritten digits-load mat file.py

does the same thing as Handwritten digits.py does. It uses MATLAB dataset (ex4data1.mat) used in https://www.coursera.org/learn/machine-learning/programming/Y54Zu/multi-class-classification-and-neural-networks assignment. 
There are 5000 training examples in ex3data1.mat, where each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at that location.

Data scraper for maxi-shop.py

does a simple data scrape from web page using BeautifulSoup library.

Recommendation for books.py

uses Surprise library. It is an easy-to-use Python scikit for recommender systems.
Installation guide is available at http://surpriselib.com/.
Dataset has been downloaded from http://www2.informatik.uni-freiburg.de/~cziegler/BX/
We first train an SVD algorithm on the whole dataset, and then predict all the ratings for the pairs (user, books) that are not in the training set. We then retrieve the top-10 prediction for each user.
Since I have memory error caused by huge used matrices, I cut the dataset and work just with first 5000 records.
The quate marks (") have been removed.
The evaluation of the algorithm is the following:
Mean RMSE: 3.3920
Mean MAE : 2.7805

