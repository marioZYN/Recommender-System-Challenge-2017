import sys
import time
import numpy as np
import scipy.sparse as sps

def sigmoid(x):

    return 1 / (1 + np.exp(-x))



class SLIM_BPR:

    def __init__(self, urm_train):

        self.urm_train = urm_train
        self.n_users = urm_train.shape[0]
        self.n_items = urm_train.shape[1]
        self.item_item = sps.csr_matrix((self.n_items, self.n_items), dtype=np.float32)