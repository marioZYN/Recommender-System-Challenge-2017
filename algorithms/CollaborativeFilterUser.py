from algorithms.Recommender import Recommender
from similarity_cython.similarity_cython.CosineSim import Cosine_Similarity
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.preprocessing import normalize

class CollaborativeFilterUser(Recommender):


    def __init__(self):
        super(CollaborativeFilterUser, self).__init__()
        self.playlist_playlist = None
        self.playlist_track = None

    def fit(self, artist_weight=0, album_weight=0, owner_weight=0):
        cos = Cosine_Similarity(self.urm.T.tocsr(), TopK=100)
        self.playlist_playlist = cos.compute_similarity().tocsr()
        # self.playlist_playlist = normalize(self.playlist_playlist, norm='l1', axis=1)
        if artist_weight != 0:
            self.add_artist(artist_weight)
        if album_weight != 0:
            self.add_album(album_weight)
        if owner_weight != 0:
            self.add_owner(owner_weight)
        self.playlist_playlist = normalize(self.playlist_playlist, norm='l1', axis=1)
        print("-- playlist_track rec generating...")
        self.playlist_track = self.playlist_playlist * self.urm
        self.playlist_track = normalize(self.playlist_track, norm='l1', axis=1)

    def add_owner(self, weight):

        print('adding owners with weight %f ...' % weight)
        playlist_final = pd.read_csv("./Data/playlists_final.csv", sep='\t')
        total_owners = sorted(list(set(playlist_final.owner)))
        row_number = len(self.playlist_unique)
        col_number = len(total_owners)

        A = sps.lil_matrix((row_number, col_number))
        count = 0
        for p in self.playlist_unique:
            row_index = self.playlist_unique.index(p)
            owner = playlist_final[playlist_final['playlist_id'] == p].iloc[0]['owner']
            col_index = total_owners.index(owner)
            A[row_index, col_index] = 1
            count += 1
            print("\r-- %d playlist completes with %d total" % (count, len(self.playlist_unique)), end='')
        print()
        cos = Cosine_Similarity(A.T.tocsr(), TopK=100)
        playlist_playlist = cos.compute_similarity()
        playlist_playlist = normalize(playlist_playlist, norm='l1', axis=1)
        self.playlist_playlist += playlist_playlist * weight

    def add_owner(self, weight):

        # adding owner does not improve the performance
        print('adding owners with weight %f ...' % weight)
        playlist_final = pd.read_csv("./Data/playlists_final.csv", sep='\t')
        total_owners = sorted(list(set(playlist_final.owner)))
        row_number = len(self.playlist_unique)
        col_number = len(total_owners)

        A = sps.lil_matrix((row_number, col_number))
        count = 0
        for p in self.playlist_unique:
            row_index = self.playlist_unique.index(p)
            owner = playlist_final[playlist_final['playlist_id'] == p].iloc[0]['owner']
            col_index = total_owners.index(owner)
            A[row_index, col_index] = 1
            count += 1
            print("\r-- %d playlist completes with %d total" % (count, len(self.playlist_unique)), end='')
        print()

        cos = Cosine_Similarity(A.T.tocsr(), TopK=500)
        playlist_playlist = cos.compute_similarity()
        playlist_playlist = normalize(playlist_playlist, norm='l1', axis=1)
        self.playlist_playlist += playlist_playlist * weight

    def add_titles(self, weight):

        print("adding titles with weight %f ..." % weight)
        playlists_final = pd.read_csv("./Data/playlists_final.csv", sep='\t')
        t1 = playlists_final[['playlist_id', 'title']]
        t1['title'] = t1['title'].str.replace(r'(\[|\])', '')

        legal_titles = set()
        count = 0
        print("-- calculating total titles...")
        for i in range(0, t1.shape[0]):
            count += 1
            print('\r-- %d playlist completes with total %d' % (count, t1.shape[0]), end='')
            temp = t1.iloc[i].title.replace(' ', '').split(',')
            if not temp[0]:
                continue
            current_titles = set(map(int, temp))
            legal_titles = legal_titles | current_titles
        print()

        total_titles = sorted(list(legal_titles))

        row_number = len(self.playlist_unique)
        col_number = len(total_titles)
        A = sps.lil_matrix((row_number, col_number))

        print("-- forming matrix...")
        count = 0
        for t in self.playlist_unique:
            row_index = self.playlist_unique.index(t)
            titles = t1[t1['playlist_id'] == t].iloc[0]['title']
            titles = titles.replace(' ', '').split(',')
            if not titles[0]:
                count += 1
                print("\r-- %d playlist completes with %d total" % (count, len(self.playlist_unique)), end='')
                continue
            titles = list(map(int, titles))
            for title in titles:
                col_index = total_titles.index(title)
                A[row_index, col_index] = 1
            count += 1
            print("\r-- %d playlist completes with %d total" % (count, len(self.playlist_unique)), end='')
        print()

        cos = Cosine_Similarity(A.T.tocsr(), TopK=500)
        playlist_playlist = cos.compute_similarity()
        playlist_playlist = normalize(playlist_playlist, norm='l1', axis=1)
        self.playlist_playlist += playlist_playlist * weight


    def add_artist(self, weight):

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

    def add_album(self, weight):

        print("adding album with weight %f ..." % weight)
        tracks_final = pd.read_csv("./Data/tracks_final.csv", sep='\t')
        total_albums = sorted(list(set(tracks_final.album)))
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

        playlist_album = self.urm * A.tocsr()
        cos = Cosine_Similarity(playlist_album.T.tocsr(), TopK=500)
        playlist_playlist = cos.compute_similarity()
        playlist_playlist = normalize(playlist_playlist, norm='l1', axis=1)
        self.playlist_playlist += playlist_playlist * weight

    def recommend(self, user_id, at=5):

        user_index = self.playlist_dic[user_id]
        user = self.playlist_playlist.getrow(user_index)
        rec = np.dot(user, self.urm)
        recommendingItems = np.asarray(rec.toarray()[0].argsort()[::-1])
        unseen_items_mask = np.in1d(recommendingItems, self.urm[user_index].indices,assume_unique=True, invert=True)
        unseen_items = recommendingItems[unseen_items_mask]
        recommended_items = unseen_items[0:at]
        return recommended_items
