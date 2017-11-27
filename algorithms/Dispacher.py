from algorithms.Recommender import Recommender
from algorithms.ContentBased import ContentBased
from algorithms.CollaborativeFilterItem import CollaborativeFilterItem
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from support import map_score


class Dispacher(Recommender):


    def __init__(self):
        super(Dispacher, self).__init__()
        self.playlist_track = None
        self.urm = None

    def evaluate(self, train, validate, test, rec1, rec2):

        playlists = validate[['playlist_id']].drop_duplicates()
        pids = list(playlists.playlist_id)
        tracks_validate = pd.merge(train[['track_id']], validate[['track_id']], on='track_id').drop_duplicates()
        tracks_test = pd.merge(train[['track_id']], test[['track_id']], on='track_id').drop_duplicates()

        legal_items_validate = set(tracks_validate.track_id)
        legal_items_test = set(tracks_test.track_id)

        count = 0
        _map = 0
        for pid in pids:
            relevant_items_validate = list(validate[validate['playlist_id'] == pid].track_id)

            recommended_items_index_1 = rec1.recommend(pid, 100)
            recommended_items_id_1 = list(map(lambda x: rec1.tracks_unique[x], recommended_items_index_1))
            recommended_items_id_1 = list(filter(lambda x: x in legal_items_validate, recommended_items_id_1))[:5]
            score_1 = map_score(recommended_items_id_1, relevant_items_validate)
            recommended_items_index_2 = rec2.recommend(pid, 100)
            recommended_items_id_2 = list(map(lambda x: rec2.tracks_unique[x], recommended_items_index_2))
            recommended_items_id_2 = list(filter(lambda x: x in legal_items_validate, recommended_items_id_2))[:5]
            score_2 = map_score(recommended_items_id_2, relevant_items_validate)

            if score_1 >= score_2:
                rec = rec1
            else:
                rec = rec2

            recommended_items_index = rec.recommend(pid, 100)
            recommended_items_id = list(map(lambda x: rec.tracks_unique[x], recommended_items_index))
            recommended_items_id = list(filter(lambda x: x in legal_items_test, recommended_items_id))[:5]
            if len(recommended_items_id) != 5:
                print("Not enough!")
                return

            relevant_items = list(test[test['playlist_id'] == pid].track_id)

            _map += map_score(recommended_items_id, relevant_items)
            count += 1
            print("\r-- %d playlist completes with total %d" % (count, len(pids)), end='', flush=True)
        print()
        _map /= count
        print()
        print("MAP = %f" % _map)




    def recommend(self, user_id, at=5):

        user_index = self.playlist_dic[user_id]
        rec = self.playlist_track.getrow(user_index)
        recommendingItems = np.asarray(rec.toarray()[0].argsort()[::-1])
        unseen_items_mask = np.in1d(recommendingItems, self.urm[user_index].indices,assume_unique=True, invert=True)
        unseen_items = recommendingItems[unseen_items_mask]
        recommended_items = unseen_items[0:at]
        return recommended_items

