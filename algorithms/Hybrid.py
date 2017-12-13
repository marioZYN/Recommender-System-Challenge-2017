from algorithms.Recommender import Recommender
from algorithms.CollaborativeFilterItem import CollaborativeFilterItem
from algorithms.CollaborativeFilterUser import CollaborativeFilterUser
from algorithms.ContentBased import ContentBased
from algorithms.ContentBasedUser import ContentBasedUser
from algorithms.BPR import BPR
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import numpy as np


class Hybrid(Recommender):

    def __init__(self):
        super(Hybrid, self).__init__()
        self.playlist_track = None
        self.urm = None
        self.rec_systems = None
        self.rec_weights = None

    def fit(self, recommender_objects):

        self.rec_systems = recommender_objects

    def get_most_rated_times(self, item_list, topk):

        return list(map(lambda x: x[0], Counter(item_list).most_common(topk)))

    def recommend(self, user_id, at=5):

        combined_rec_item_indexes = []
        for rec_sys in self.rec_systems:
            rec = rec_sys.recommend(user_id, at)
            combined_rec_item_indexes += rec

        return self.get_most_rated_times(combined_rec_item_indexes, topk=at)


if __name__ == '__main__':

    from support.utility import read_data
    from support.utility import train_test_split
    from support.utility import train_validate_test_split
    from sklearn.preprocessing import normalize
    import time

    start = time.time()
    print("reading data")
    data = read_data(sample_frac=0.9, only_target=False)
    print("train, test splitting")
    (train, test1, test2) = train_validate_test_split(data)

    # print("training cfi")
    # cfi = CollaborativeFilterItem()
    # cfi.setup(train)
    # cfi.fit()

    # print("training cfu")
    # cfu = CollaborativeFilterUser()
    # cfu.setup(train, test)
    # cfu.fit()


    print("training cb item")
    cb_item = ContentBased()
    cb_item.setup(train)
    cb_item.fit(artist=15, album=15, owner=2, tag=1, history=5)
    #
    print("training cb user")
    cb_user = ContentBasedUser()
    cb_user.setup(train)
    cb_user.fit(artist=1)
    

    # print("training bpr")
    # bpr = BPR()
    # bpr.setup(train, test)
    # bpr.fit(n_iteration=15, topK=200, learning_rate=1e-3, lambda_i=0.0001, lambda_j=0.0001)

    # print("adding")
    # cfi.playlist_track = normalize(cfi.playlist_track, norm='l2', axis=1)
    # cb_item.playlist_track = normalize(cb_item.playlist_track, norm='l2', axis=1)
    # cfi.playlist_track += cb_item.playlist_track
    # print("evaluating test1")
    # cfi.evaluate_result(train, test1)

    # print("evaluating test2")
    # cfi.evaluate_result(train, test2)
    #
    # cfi.gen_result("../results/hybrid_cb_cfi_v2.csv")


    # rec_sys = [cfi,cfu]
    # for i in range(1,len(rec_sys)+1):
    #     rec = rec_sys[0:i]
    #     print("training hybrid with %d"%i)
    #     hybrid = Hybrid()
    #     hybrid.setup(train,test)
    #     hybrid.fit(rec)
    #     print("evaluating")
    #     hybrid.evaluate_result(train, test)
    print("total time is {:.2f} minutes".format((time.time() - start) / 60))


