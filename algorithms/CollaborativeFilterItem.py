from algorithms.Recommender import Recommender
from similarity_cython.similarity_cython.CosineSim import Cosine_Similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import scipy.sparse as sps


class CollaborativeFilterItem(Recommender):

    def __init__(self):

        super().__init__()
        self.track_track = None
        self.playlist_track = None

    def fit(self, artist_weight=0, album_weight=0,tag_weight=0, combine_weight=0,artist=0, album=0, topk=100):


        print("-- cosing cfi with topK=%d..."%topk)
        # track_track = cosine_similarity(self.urm.T.tocsr(), dense_output=False)
        # self.track_track = self.pruneTopK(track_track, topK=topk).T.tocsr()
        # self.add_tags(1)
        # temp = sps.vstack([self.urm.tocoo(), self.tags.T.tocoo()])
        cos = Cosine_Similarity(self.urm.tocsr(), TopK=topk)
        self.track_track = cos.compute_similarity()
        self.track_track = normalize(self.track_track, norm='l1', axis=1)
        if artist_weight != 0:
            print("-- adding artist with weight %f"%artist_weight)
            self.add_artist(artist_weight)
        if album_weight != 0:
            print("-- adding album with weight %f"%album_weight)
            self.add_album(album_weight)
        if combine_weight !=0:
            print("-- vstacking artist %f album %f combine %f"%(artist, album, combine_weight))
            self.vstack(artist_weight=artist, album_weight=album, combine_weight=combine_weight, topk=topk)
        if tag_weight != 0:
            self.add_tags(tag_weight)
        print("-- playlist_track rec generating...")
        self.playlist_track = self.urm * self.track_track
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

        # cos = Cosine_Similarity(A.T.tocsr(), TopK=300)
        # track_track = cos.compute_similarity()
        # track_track = normalize(track_track, norm='l1', axis=1)
        track_track = cosine_similarity(A.tocsr(), dense_output=False)
        track_track = self.pruneTopK(track_track, topK=3000)
        self.track_track += track_track * weight

    def add_album(self, weight):

        tracks_final = pd.read_csv("./Data/tracks_final.csv", sep='\t')
        temp = pd.DataFrame({'track_id': self.tracks_unique})
        temp = pd.merge(tracks_final, temp, on='track_id')
        total_albums = sorted(list(set(temp.album)))
        total_albums.remove('[]')
        total_albums.remove('[None]')
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
                continue
            col_index = total_albums.index(album)
            A[row_index, col_index] = 1
            count += 1
        cos = Cosine_Similarity(A.T.tocsr(), TopK=300)
        track_track = cos.compute_similarity()
        track_track = normalize(track_track, norm='l1', axis=1)
        self.track_track += track_track * weight

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
        # track_track = self.pruneTopK(track_track, topK=topk).T.tocsr()
        self.track_track += track_track * combine_weight

    def add_tags(self, weight):

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

        self.tags = A.tocsr()


    def recommend(self, user_id, at=5):

        user_index = self.playlist_dic[user_id]
        rec = self.playlist_track.getrow(user_index)
        recommendingItems = np.asarray(rec.toarray()[0].argsort()[::-1])
        unseen_items_mask = np.in1d(recommendingItems, self.urm[user_index].indices,assume_unique=True, invert=True)
        unseen_items = recommendingItems[unseen_items_mask]
        recommended_items = unseen_items[0:at]
        return recommended_items
