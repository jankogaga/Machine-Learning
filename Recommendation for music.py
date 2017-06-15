#dataset https://grouplens.org/datasets/hetrec-2011/

from collections import defaultdict

from surprise import SVD
from surprise import Dataset
import os
from surprise import Reader

from surprise import evaluate, print_perf

def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.
    
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''
    
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n


# First train an SVD algorithm on the movielens dataset.
file_path = os.path.expanduser('./input/user_artists.dat')
reader = Reader(line_format='user item rating', sep='\t', rating_scale=(1, 20000))
data = Dataset.load_from_file(file_path, reader=reader)

trainset = data.build_full_trainset()
algo = SVD()
algo.train(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=10)

# Print the recommended items for each user
#for uid, user_ratings in top_n.items():
#    print(uid, [iid for (iid, _) in user_ratings])

#E.g.: for the user 1150, recommended items are:
for uid, user_ratings in top_n.items():
	if uid == '1150':
		print(uid, [iid for (iid, _) in user_ratings])

# Evaluate performances of our algorithm on the dataset.
data.split(n_folds=3)
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print_perf(perf)

