from algorithms.Recommender import Recommender
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import scipy.sparse as sps
import numpy as np

class ContentBased(Recommender):

    def __init__(self):

        super().__init__()
        self.track_track = None
        self.playlist_track = None

    def fit(self, artist=0, album=0):

        print("-- creating attributes")
        attributes = []
        if artist != 0:
            attributes.append(self.generate_track_artist_matrix(artist))
        if album != 0:
            attributes.append(self.generate_track_album_matrix(album))
        print("-- combining attributes")
        result = attributes[0]
        for attribute in attributes[1:]:
            result = sps.hstack([result.tocoo(), attribute.tocoo()])
        print("-- generating cosine similarity matrix")
        self.track_track = cosine_similarity(result.tocsr(), dense_output=False)
        print("-- pruning and keep only top 500")
        self.track_track = self.pruneTopK(self.track_track, topK=500)
        print("-- generating prediction matrix")
        self.playlist_track = self.urm.tocsr() * self.track_track.tocsr()

    def generate_track_artist_matrix(self, weight):

        print("* generating track_artist_matrix with weight {:.1f}".format(weight))

        # reading data and perform prune
        tracks_final = pd.read_csv("../data/tracks_final.csv", sep='\t')
        temp = pd.DataFrame({'track_id': self.tracks_unique})
        track_artist_df = pd.merge(temp, tracks_final[['track_id', 'artist_id']], on='track_id')
        artists = list(set(track_artist_df.artist_id))
        track_id_artist_id_dic = dict(zip(list(track_artist_df.track_id), list(track_artist_df.artist_id)))
        artist_id_index_dic = dict(zip(artists, list(np.arange(0, len(artists)))))

        # generating matrix
        ratinglist = [weight] * len(self.tracks_unique)
        row_indices = [x for x in range(0, len(self.tracks_unique))]
        col_indices = list(map(lambda x: artist_id_index_dic[track_id_artist_id_dic[x]], self.tracks_unique))
        track_artist_matrix = sps.coo_matrix((ratinglist, (row_indices, col_indices)))

        return track_artist_matrix

    def generate_track_album_matrix(self, weight):

        print("* generating track_album_matrix with weight {:.1f}".format(weight))

        # reading data and perform prune
        tracks_final = pd.read_csv("../data/tracks_final.csv", sep='\t')
        temp = pd.DataFrame({'track_id': self.tracks_unique})
        track_album_df = pd.merge(temp, tracks_final[['track_id', 'album']], on='track_id')
        albums = list(set(track_album_df.album))
        track_id_album_dic = dict(zip(list(track_album_df.track_id), list(track_album_df.album)))
        album_index_dic = dict(zip(albums, list(np.arange(0, len(albums)))))

        # generating matrix
        ratinglist = [0 if ( track_id_album_dic[x] == '[]' or track_id_album_dic[x] == '[]') else weight for x in self.tracks_unique]
        row_indices = [x for x in range(0, len(self.tracks_unique))]
        col_indices = list(map(lambda x: album_index_dic[track_id_album_dic[x]], self.tracks_unique))
        track_album_matrix = sps.coo_matrix((ratinglist, (row_indices, col_indices)))

        return track_album_matrix

    def recommend(self, user_id, at=5):

        user_index = self.playlist_dic[user_id]
        rec = self.playlist_track.tocsr().getrow(user_index)
        recommendingItems = np.asarray(rec.toarray()[0].argsort()[::-1])
        unseen_items_mask = np.in1d(recommendingItems, self.urm[user_index].indices,assume_unique=True, invert=True)
        unseen_items = recommendingItems[unseen_items_mask]
        recommended_items = unseen_items[0:at]
        return recommended_items



if __name__ == "__main__":

    from support.utility import read_data
    from support.utility import train_test_split
    import time

    start = time.time()
    print("reading data")
    data = read_data(sample_frac=0.9)
    print("train, test splitting")
    (train, test) = train_test_split(data, 5)

    print("training cb")
    cb = ContentBased()
    cb.setup(train)
    cb.fit(artist=1, album=1)
    print("evaluating")
    cb.evaluate_result(train, test)

    print("total time is {:.2f} minutes".format((time.time() - start) / 60))