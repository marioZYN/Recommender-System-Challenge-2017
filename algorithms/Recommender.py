import pandas as pd
import numpy as np
import scipy.sparse as sps
import random
from support.utility import map_score


class Recommender(object):

    """Abstract recommender class"""

    def __init__(self):

        self.playlist_unique = None
        self.tracks_unique = None
        self.playlist_dic = None
        self.tracks_dic = None
        self.playlist_index = None
        self.tracks_index = None
        self.playlist_track = None
        self.urm = None


    def setup(self, train):

        train_sorted = train.sort_values(['playlist_id', 'track_id'])
        playlist_unique = list(train_sorted['playlist_id'].unique())
        tracks_unique = list(train_sorted['track_id'].unique())

        playlist_dic = dict(zip(playlist_unique, list(np.arange(0, len(playlist_unique)))))
        tracks_dic = dict(zip(tracks_unique, list(np.arange(0, len(tracks_unique)))))

        playlist_index = list(map(lambda x: playlist_dic[x], list(train_sorted['playlist_id'])))
        tracks_index = list(map(lambda x: tracks_dic[x], list(train_sorted['track_id'])))
        rating_list = [1] * train_sorted.shape[0]
        urm = sps.coo_matrix((rating_list, (playlist_index, tracks_index)))


        self.playlist_unique = playlist_unique
        self.tracks_unique = tracks_unique
        self.playlist_dic = playlist_dic
        self.tracks_dic = tracks_dic
        self.playlist_index = playlist_index
        self.tracks_index = tracks_index
        self.urm = urm.tocsr()


    def recommend(self, user_id, at=5):

        user_index = self.playlist_dic[user_id]
        rec = self.playlist_track.tocsr().getrow(user_index)
        recommendingItems = np.asarray(rec.toarray()[0].argsort()[::-1])
        unseen_items_mask = np.in1d(recommendingItems, self.urm[user_index].indices, assume_unique=True, invert=True)
        unseen_items_index = recommendingItems[unseen_items_mask]
        recommended_items = unseen_items_index[0:at]
        return recommended_items

    def gen_result(self, path):

        target_playlist = pd.read_csv('../data/target_playlists.csv', sep='\t')
        target_tracks = pd.read_csv('../data/target_tracks.csv', sep='\t')
        result = target_playlist.copy().sort_values(['playlist_id'])
        result = result.reset_index(drop=True)
        pids = list(result.playlist_id)
        legal_items = set(target_tracks.track_id)
        count = 0
        for pid in pids:
            recommended_items_index = self.recommend(pid, 100)
            recommended_items_id = list(map(lambda x: self.tracks_unique[x], recommended_items_index))
            recommended_items_id = list(filter(lambda x: x in legal_items, recommended_items_id))[:5]
            if len(recommended_items_id) != 5:
                print("Not enough!")
                return
            result.set_value(index=count, col='track_ids', value=' '.join(str(x) for x in recommended_items_id))
            count += 1
            print("\r%d playlist completes..." % count, end='', flush=True)
        print()
        print("finished")
        result.to_csv(path, index=False)

    def evaluate_result(self, train, test):

        playlists = test[['playlist_id']].drop_duplicates()
        tracks = pd.merge(train[['track_id']], test[['track_id']], on='track_id').drop_duplicates()

        pids = list(playlists.playlist_id)
        legal_items = set(tracks.track_id)

        count = 0
        _map = 0.0
        for pid in pids:

            recommended_items_index = self.recommend(pid, 200)
            recommended_items_id = list(map(lambda x: self.tracks_unique[x], recommended_items_index))
            recommended_items_id = list(filter(lambda x: x in legal_items, recommended_items_id))[:5]

            relevant_items = list(test[test['playlist_id'] == pid].track_id)

            _map += map_score(recommended_items_id, relevant_items)
            count += 1
            print("\r%d playlist completes with total %d" %(count, len(pids)), end='', flush=True)
        print()

        _map /= count
        print()
        print("MAP = %f" % _map)
