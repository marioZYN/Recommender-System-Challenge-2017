from algorithms.CollaborativeFilterItem import CollaborativeFilterItem
import numpy as np
from math import exp
import random
import time

def sigmoid(gamma):
  if gamma < 0:
    return 1 - 1/(1 + exp(gamma))
  else:
    return 1/(1 + exp(-gamma))

class Sampler():

    def __init__(self, sample_neg_item_empirically):

        self.sample_neg_item_empirically = sample_neg_item_empirically

    def set_up(self, data, max_samples=None):

        self.data = data
        self.n_users, self.n_items = data.shape
        self.max_samples = max_samples

    def uniform_user(self):

        return random.randint(0, self.n_users-1)

    def sample_user(self):

        return self.uniform_user()

    def random_item(self):

        # sample an item uniformly or from the empirical distribution
        if self.sample_neg_item_empirically:
            u = self.uniform_user()
            i = random.choice(self.data[u].indices)
        else:
            i = random.randint(0, self.n_items-1)

        return i

    def sample_negative_item(self, user_items):

        j = self.random_item()
        while j in user_items:
            j = self.random_item()

        return j

    def num_samples(self, n):

        if self.max_samples is None:
            return n
        else:
            return min(n, self.max_samples)


class UniformUserUniformItem(Sampler):

    def generate_samples(self, data, max_samples=None):

        self.set_up(data, max_samples)
        for _ in range(self.num_samples(self.data.nnz)):
            u = self.uniform_user()
            i = random.choice(self.data[u].indices)
            j = self.sample_negative_item(self.data[u].indices)
            yield u,i,j

class MF_BPR():

    def __init__(self, n_factors=50, learning_rate=0.05,
                 bias_reg=1.0,
                 user_reg=0.0025,
                 pos_item_reg=0.0025,
                 neg_item_reg=0.00025,
                 update_neg_item_factors=True):

        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.bias_reg = bias_reg
        self.user_reg = user_reg
        self.pos_item_reg = pos_item_reg
        self.neg_item_reg = neg_item_reg
        self.update_neg_item_factors = update_neg_item_factors

    def set_up(self, data):

        self.data = data
        self.n_users, self.n_items = data.shape
        self.item_bias = np.zeros(self.n_items)
        self.user_factors = np.random.random_sample((self.n_users, self.n_factors))
        self.item_factors = np.random.random_sample((self.n_items, self.n_factors))

        self.create_loss_samples()

    def create_loss_samples(self):

        # apply rule of thumb to decide num samples over which to compute loss
        num_loss_samples = int(100 * self.n_users**0.5)
        print("sampling {0} <user, item i, item j> triples...".format(num_loss_samples))
        sampler = UniformUserUniformItem(True)
        self.loss_samples = [t for t in sampler.generate_samples(self.data, num_loss_samples)]

    def loss(self):

        ranking_loss = 0
        for u, i, j in self.loss_samples:
            x = self.predict(u, i) - self.predict(u, j)
            ranking_loss += sigmoid(x)

        return ranking_loss

    def predict(self, u, i):

        return self.item_bias[i] + np.dot(self.user_factors[u], self.item_factors[i])


    def train(self, data, sampler, n_iters):

        self.set_up(data)
        print("initial loss = {0}".format(self.loss()))
        for it in range(n_iters):
            print("starting iteration {0}".format(it))
            start = time.time()
            count = 0
            for u, i, j in sampler.generate_samples(self.data):
                self.update_factors(u, i, j)
                print("\r%d sample compeles"%count, end='', flush=True)
                count += 1
            print()
            print("iteration {}: loss = {}, used time {:.2f} sec".format(it, self.loss(), time.time()-start))

    def update_factors(self, u, i, j, update_u=True, update_i=True):

        update_j = self.update_neg_item_factors
        x = self.item_bias[i] - self.item_bias[j] + np.dot(self.user_factors[u,:], self.item_factors[i,:] - self.item_factors[j,:])
        z = sigmoid(x)

        # update bias term
        if update_i:
            d = z - self.bias_reg * self.item_bias[i]
            self.item_bias[i] += self.learning_rate * d

        if update_j:
            d = -z - self.bias_reg * self.item_bias[j]
            self.item_bias[j] += self.learning_rate * d

        if update_u:
            d = (self.item_factors[i,:] - self.item_factors[j]) * z - self.user_reg * self.user_factors[u,:]
            self.user_factors[u, :] += self.learning_rate * d

        if update_i:
            d = self.user_factors[u,:]*z - self.pos_item_reg * self.item_factors[i,:]
            self.item_factors[i,:] += self.learning_rate * d

        if update_j:
            d = -self.user_factors[u,:] * z - self.neg_item_reg * self.item_factors[j,:]
            self.item_factors[j,:] += self.learning_rate * d


if __name__ == '__main__':

    from support.utility import read_data
    from support.utility import train_test_split
    cfi = CollaborativeFilterItem()
    print("reading data")
    data = read_data(sample_frac=0.9)
    cfi.setup(data.sample(frac=0.1))
    data = cfi.urm.tocsr()

    model = MF_BPR(n_factors=50, learning_rate=1e-3)

    sampler = UniformUserUniformItem(sample_neg_item_empirically=True)
    model.train(data, sampler, n_iters=10)
