from algorithms.Recommender import Recommender
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
import time

class ImplicitMF(Recommender):

    def __init__(self):

        super().__init__()
        self.n_users = None
        self.n_items = None
        self.n_factors = None
        self.n_iterations = None
        self.reg = None
        self.user_vectors = None
        self.item_vectors = None
        self.playlist_track = None

    def fit(self, n_factors=40, n_iterations=40, reg=0.8):

        self.n_users = self.urm.shape[0]
        self.n_items = self.urm.shape[1]
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg = reg

        # initialize latent factor matrix
        self.user_vectors = np.random.normal(size=(self.n_users, self.n_factors))
        self.item_vectors = np.random.normal(size=(self.n_items, self.n_factors))

        for i in range(self.n_iterations):
            start = time.time()
            print("solving for user vectors")
            self.user_vectors = self.iteration(user_flag=True, fixed_vecs=self.item_vectors)
            print("sovling for item vectors")
            self.item_vectors = self.iteration(user_flag=False, fixed_vecs=self.user_vectors)
            end = time.time()
            print("iteration {} finished in {:.2f} minutes".format(i, (end-start)/60))


    def iteration(self, user_flag, fixed_vecs):

        num_solve = self.n_users if user_flag else self.n_items
        YTY = fixed_vecs.T.dot(fixed_vecs)

        lambda_eye = self.reg * sps.eye(self.n_factors)
        solve_vecs = np.zeros((num_solve, self.n_factors))

        start = time.time()
        for i in range(num_solve):
            if user_flag:
                pu = self.urm.tocsr()[i, :].toarray().squeeze()
            else:
                pu = self.urm.tocsc()[:, i].toarray().squeeze()
            YTpu = fixed_vecs.T.dot(pu.reshape(len(pu), 1))

            xu = solve(YTY + lambda_eye, YTpu)
            solve_vecs[i] = xu.ravel()

            if i % 10000 == 0 and i != 0:
                print("solved {} vecs in {:.2f} seconds".format(i, time.time() - start))

        return solve_vecs

    def recommend(self, user_id, at=5):

        user_index = self.playlist_dic[user_id]
        user_f = self.user_vectors[user_index]
        rec = user_f.dot(self.item_vectors.T)
        recommendingItems = rec.argsort().squeeze()[::-1]
        unseen_items_mask = np.in1d(recommendingItems, self.urm[user_index].indices, assume_unique=True, invert=True)
        unseen_items = recommendingItems[unseen_items_mask]
        recommended_items = unseen_items[0:at]
        return recommended_items


if __name__ == "__main__":

    from support.utility import read_data
    from support.utility import train_test_split

    start = time.time()
    print("reading data")
    data = read_data(sample_frac=0.9, only_target=True)
    print("train, test splitting")
    (train, test) = train_test_split(data, 5)

    imf = ImplicitMF()
    imf.setup(train)
    imf.fit(n_factors=800, n_iterations=20, reg=0.1)
    imf.evaluate_result(train, test)

    print("total time is {:.2f} minutes".format((time.time() - start) / 60))
