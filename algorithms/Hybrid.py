from algorithms.Recommender import Recommender
from algorithms.ContentBased import ContentBased
from algorithms.CollaborativeFilterItem import CollaborativeFilterItem
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

    def fit(self, recommender_objects, weights):

        # for (index,rec) in enumerate(recommender_objects):
        #     if self.playlist_track == None:
        #         self.playlist_track = weights[index] * rec.playlist_track
        #     else:
        #         self.playlist_track += weights[index] * rec.playlist_track

        self.rec_systems = recommender_objects
        self.rec_weights = weights

    def recommend(self, user_id, at=5):

        user_index = self.playlist_dic[user_id]
        rec = self.playlist_track.getrow(user_index)
        recommendingItems = np.asarray(rec.toarray()[0].argsort()[::-1])
        unseen_items_mask = np.in1d(recommendingItems, self.urm[user_index].indices,assume_unique=True, invert=True)
        unseen_items = recommendingItems[unseen_items_mask]
        recommended_items = unseen_items[0:at]
        return recommended_items
