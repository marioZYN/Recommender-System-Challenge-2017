from algorithms.Recommender import Recommender
from sklearn.metrics.pairwise import cosine_similarity
from similarity_cython.similarity_cython.CosineSim import Cosine_Similarity
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import scipy.sparse as sps


class CollaborativeFilterItem(Recommender):

    def __init__(self):

        super().__init__()
        self.track_track = None
        self.playlist_track = None

    def fit(self):

        print("-- generating profile")
        track_artist_matrix = self.generate_track_artist_matrix(1)
        track_album_matrix = self.generate_track_album_matrix(1)
        track_attribute_matrix = sps.hstack([track_artist_matrix.tocoo(), track_album_matrix.tocoo()])
        print("-- caluclating item-item simislarity matrix")
        self.track_track = cosine_similarity(self.urm.T.tocsr(), dense_output=False)
        self.track_track += cosine_similarity(track_attribute_matrix.tocsr(), dense_output=False)

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

    def generate_track_owner_matrix(self, weight):

        print("* generating playlist_owner_matrix with weight {:.1f}".format(weight))
        # reading data and perform prune
        playlist_final = pd.read_csv("../data/playlists_final.csv",sep='\t')
        temp = pd.DataFrame({'playlist_id': self.playlist_unique})
        playlist_owner = pd.merge(temp, playlist_final, on='playlist_id', how='left')[['playlist_id', 'owner']]
        total_owners = list(set(playlist_owner.owner))
        owner_dic = dict(zip(total_owners, list(np.arange(0, len(total_owners)))))
        playlist_owner_dic = dict(zip(list(playlist_owner.playlist_id), list(playlist_owner.owner)))

        # generating matrix
        ratingList = [weight] * len(self.playlist_unique)
        row_indices = [x for x in range(0, len(self.playlist_unique))]
        col_indices = [owner_dic[playlist_owner_dic[x]] for x in self.playlist_unique]
        playlist_owner_matrix = sps.coo_matrix((ratingList, (row_indices, col_indices)))
        track_owner_matrix = self.urm.T.tocsr() * playlist_owner_matrix

        return track_owner_matrix


    def add_tags(self, weight):

        tracks_final = pd.read_csv("./data/tracks_final.csv", sep='\t')
        t1 = tracks_final[['track_id', 'tags']]
        t2 = pd.DataFrame(self.tracks_unique, columns=['track_id'])
        t1 = pd.merge(t1, t2, on='track_id')
        t1['tags'] = t1['tags'].str.replace(r'(\[|\])', '')

        legal_tags = set()
        count = 0
        print("-- calculating total tags...")
        for i in range(0, t1.shape[0]):
            count += 1
            print('\r-- %d tracks completes with total %d' % (count, t1.shape[0]), end='')
            temp = t1.iloc[i].tags.replace(' ', '').split(',')
            if not temp[0]:
                continue
            current_tags = set(map(int, temp))
            legal_tags = legal_tags | current_tags
        print()

        total_tags = sorted(list(legal_tags))

        row_number = len(self.tracks_unique)
        col_number = len(total_tags)
        A = sps.lil_matrix((row_number, col_number))

        print("-- forming matrix...")
        count = 0
        for t in self.tracks_unique:
            row_index = self.tracks_unique.index(t)
            tags = t1[t1['track_id'] == t].iloc[0]['tags']
            tags = tags.replace(' ', '').split(',')
            if not tags[0]:
                count += 1
                print("\r-- %d tracks completes with %d total" % (count, len(self.tracks_unique)), end='')
                continue
            tags = list(map(int, tags))
            for tag in tags:
                col_index = total_tags.index(tag)
                A[row_index, col_index] = 1
            count += 1
            print("\r-- %d tracks completes with %d total" % (count, len(self.tracks_unique)), end='')
        print()

        self.tags = A.tocsr()


    def recommend(self, user_id, at=5):

        user_index = self.playlist_dic[user_id]
        user = self.urm[user_index]
        rec = np.dot(user, self.track_track)
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

    print("training cfi")
    cfi = CollaborativeFilterItem()
    cfi.setup(train)
    cfi.fit()
    print("evaluating")
    cfi.evaluate_result(train, test)

    print("total time is {:.2f} minutes".format((time.time() - start) / 60))