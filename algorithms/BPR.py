from similarity_cython.SLIM_BPR.Cython.SLIM_BPR_Cython  import SLIM_BPR_Cython
from similarity_cython.similarity_cython.CosineSim import Cosine_Similarity
from algorithms.Recommender import Recommender
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import scipy.sparse as sps

class BFR(Recommender):

    def __init__(self):

        super().__init__()
        self.track_track = None
        self.playlist_track = None

    def fit(self,n_iteration, topK, learning_rate, lambda_i, lambda_j,combine_weight=[1,1], artist_weight=0, album_weight=0, artist_cfu_weight=0, vstack_weight=0):

        #  track-track part
        recommender = SLIM_BPR_Cython(self.urm.tocsr(), recompile_cython=False, positive_threshold=0, sparse_weights=True)
        recommender.fit(epochs=n_iteration, validate_every_N_epochs=100, URM_test=None, topK=topK,
                        batch_size=1, sgd_mode='sgd', learning_rate=learning_rate, lambda_i=lambda_i, lambda_j=lambda_j)
        self.track_track = recommender.S
        # self.track_track = normalize(self.track_track, norm='l1', axis=1)
        if artist_weight != 0:
            self.add_artist(artist_weight)
        if album_weight !=0:
            self.add_album(album_weight)
        if vstack_weight != 0:
            self.vstack(artist_weight=1, album_weight=1, combine_weight=1,topk=300)
        # self.playlist_track = self.urm * self.track_track
        # self.playlist_track = normalize(self.playlist_track, norm='l1', axis=1)


        # playlist-playlist part
        recommender = SLIM_BPR_Cython(self.urm.T.tocsr(), recompile_cython=False, positive_threshold=0,
                                      sparse_weights=True)
        recommender.fit(epochs=n_iteration, validate_every_N_epochs=100, URM_test=None, topK=topK,
                        batch_size=1, sgd_mode='sgd', learning_rate=learning_rate, lambda_i=lambda_i, lambda_j=lambda_j)
        self.playlist_playlist = recommender.S
        if artist_cfu_weight != 0:
            self.add_artist_cfu(artist_cfu_weight)

        # self.playlist_playlist = normalize(self.playlist_playlist, norm='l1', axis=1)
        # # self.playlist_track = self.playlist_playlist.tocsr() * self.urm
        # # self.playlist_track = normalize(self.playlist_track, norm='l1', axis=1)

        self.playlist_track = self.urm * self.track_track * combine_weight[0] + self.playlist_playlist * self.urm * combine_weight[1]
        self.playlist_track = normalize(self.playlist_track, norm='l1', axis=1)

    def add_artist(self, weight):

        tracks_final = pd.read_csv("./Data/tracks_final.csv", sep='\t')
        temp = pd.DataFrame({'track_id': self.tracks_unique})

        total_artists = list(set(tracks_final.artist_id))
        artist_dic = dict(zip(total_artists, list(np.arange(0, len(total_artists)))))

        track_artists = pd.merge(temp, tracks_final, on='track_id', how='left')[['track_id', 'artist_id']]
        ratingList = [1] * len(self.tracks_unique)
        row_indices = [x for x in range(0, len(self.tracks_unique))]
        col_indices = list(map(lambda x: artist_dic[track_artists[track_artists['track_id'] == x].iloc[0]['artist_id']],
                               self.tracks_unique))
        A = sps.coo_matrix((ratingList, (row_indices, col_indices)))

        cos = Cosine_Similarity(A.T.tocsr(), TopK=500)
        track_track = cos.compute_similarity()
        track_track = normalize(track_track, norm='l1', axis=1)
        self.track_track += track_track * weight

    def add_album(self, weight):

        print("adding album with weight %f ..." % weight)
        tracks_final = pd.read_csv("./Data/tracks_final.csv", sep='\t')
        temp = pd.DataFrame({'track_id': self.tracks_unique})
        temp = pd.merge(tracks_final, temp, on='track_id')
        total_albums = sorted(list(set(temp.album)))
        total_albums.remove('[]')
        # total_albums.remove('[None]')
        legal_albums = set(total_albums)
        row_number = len(self.tracks_unique)
        col_number = len(total_albums)

        A = sps.lil_matrix((row_number, col_number))
        count = 0
        for t in self.tracks_unique:
            row_index = self.tracks_unique.index(t)
            album = tracks_final[tracks_final['track_id'] == t].iloc[0]['album']
            if album not in legal_albums:
                count += 1
                print("\r-- %d track completes with %d total" % (count, len(self.tracks_unique)), end='')
                continue
            col_index = total_albums.index(album)
            A[row_index, col_index] = 1
            count += 1
            print("\r-- %d track completes with %d total" % (count, len(self.tracks_unique)), end='')
        print()

        cos = Cosine_Similarity(A.T.tocsr(), TopK=100)
        track_track = cos.compute_similarity()
        track_track = normalize(track_track, norm='l1', axis=1)
        self.track_track += track_track * weight

    def recommend(self, user_id, at=5):

        user_index = self.playlist_dic[user_id]
        rec = self.playlist_track.getrow(user_index)
        recommendingItems = np.asarray(rec.toarray()[0].argsort()[::-1])
        unseen_items_mask = np.in1d(recommendingItems, self.urm[user_index].indices,assume_unique=True, invert=True)
        unseen_items = recommendingItems[unseen_items_mask]
        recommended_items = unseen_items[0:at]
        return recommended_items

    def vstack(self, artist_weight=2, album_weight=1, combine_weight=1, topk=100):

        # Artist part
        tracks_final = pd.read_csv("./Data/tracks_final.csv", sep='\t')
        temp = pd.DataFrame({'track_id': self.tracks_unique})

        total_artists = list(set(tracks_final.artist_id))
        artist_dic = dict(zip(total_artists, list(np.arange(0, len(total_artists)))))

        track_artists = pd.merge(temp, tracks_final, on='track_id', how='left')[['track_id', 'artist_id']]
        ratingList = [artist_weight] * len(self.tracks_unique)
        row_indices = [x for x in range(0, len(self.tracks_unique))]
        col_indices = list(map(lambda x: artist_dic[track_artists[track_artists['track_id'] == x].iloc[0]['artist_id']],
                               self.tracks_unique))
        artist_matrix = sps.coo_matrix((ratingList, (row_indices, col_indices)))

        # album part
        tracks_final = pd.read_csv("./Data/tracks_final.csv", sep='\t')
        temp = pd.DataFrame({'track_id': self.tracks_unique})
        temp = pd.merge(tracks_final, temp, on='track_id')
        total_albums = sorted(list(set(temp.album)))
        total_albums.remove('[]')
        total_albums.remove('[None]')
        legal_albums = set(total_albums)
        row_number = len(self.tracks_unique)
        col_number = len(total_albums)

        album_matrix = sps.lil_matrix((row_number, col_number))
        for t in self.tracks_unique:
            row_index = self.tracks_unique.index(t)
            album = tracks_final[tracks_final['track_id'] == t].iloc[0]['album']
            if album not in legal_albums:
                continue
            col_index = total_albums.index(album)
            album_matrix[row_index, col_index] = album_weight

        # vstack
        artist_album_matrix = sps.hstack([artist_matrix, album_matrix.tocoo()])

        # cos similarity
        cos = Cosine_Similarity(artist_album_matrix.T.tocsr(), TopK=topk)
        track_track = cos.compute_similarity()
        track_track = normalize(track_track, norm='l1', axis=1)

        # track_track = cosine_similarity(artist_album_matrix.tocsr(), dense_output=False)
        # track_track.setdiag(0)
        # track_track.eliminate_zeros()
        self.track_track += track_track * combine_weight

    def add_artist_cfu(self, weight):

        tracks_final = pd.read_csv("./Data/tracks_final.csv", sep='\t')
        temp = pd.DataFrame({'track_id': self.tracks_unique})
        track_artists = pd.merge(temp, tracks_final, on='track_id', how='left')[['track_id', 'artist_id']]
        ratingList = [1] * len(self.tracks_unique)
        row_indices = [x for x in range(0, len(self.tracks_unique))]
        col_indices = list(
            map(lambda x: track_artists[track_artists['track_id'] == x].iloc[0]['artist_id'], self.tracks_unique))
        track_artist = sps.coo_matrix((ratingList, (row_indices, col_indices)))
        playlist_artist = self.urm * track_artist.tocsr()
        cos = Cosine_Similarity(playlist_artist.T.tocsr(), TopK=100)
        playlist_playlist = cos.compute_similarity()
        playlist_playlist = normalize(playlist_playlist, norm='l1', axis=1)
        self.playlist_playlist += playlist_playlist * weight

    def gen_tags(self, weight):

        tracks_final = pd.read_csv("./Data/tracks_final.csv", sep='\t')
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

        return A.T.tocsr()