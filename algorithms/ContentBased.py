from algorithms.Recommender import Recommender
from similarity_cython.similarity_cython.CosineSim import Cosine_Similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pandas as pd
import scipy.sparse as sps
import numpy as np

class ContentBased(Recommender):

    def __init__(self):

        super(ContentBased, self).__init__()
        self.track_track = None
        self.playlist_track = None

    def fit(self, artist_flag=False, artist_weight=0, album_flag=False, album_weight=0, tag_flag=False, tag_weight=0):

        if artist_flag:
            self.add_artist(artist_weight)
        if album_flag:
            self.add_album(album_weight)
        if tag_flag:
            self.add_tags(tag_weight)

        self.track_track = normalize(self.track_track, norm='l1', axis=1)


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

        cos = Cosine_Similarity(A.T.tocsr(), TopK=50)
        track_track = cos.compute_similarity()
        track_track = normalize(track_track, norm='l1', axis=1)


        if self.track_track == None:
            self.track_track =  track_track * weight
        else:
            self.track_track += track_track * weight

    def add_album(self, weight):

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
        for t in self.tracks_unique:
            row_index = self.tracks_unique.index(t)
            album = tracks_final[tracks_final['track_id'] == t].iloc[0]['album']
            if album not in legal_albums:
                continue
            col_index = total_albums.index(album)
            A[row_index, col_index] = 1

        cos = Cosine_Similarity(A.T.tocsr(), TopK=50)
        track_track = cos.compute_similarity()
        track_track = normalize(track_track, norm='l1', axis=1)

        if self.track_track == None:
            self.track_track = track_track * weight
        else:
            self.track_track += track_track * weight



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

        cos = Cosine_Similarity(A.T.tocsr(),TopK=50)
        track_track = cos.compute_similarity()
        track_track = normalize(track_track, norm='l1', axis=1)

        if self.track_track == None:
            self.track_track = track_track * weight
        else:
            self.track_track += track_track * weight

    def recommend(self, user_id, at=5):

        user_index = self.playlist_dic[user_id]
        rec = self.playlist_track.getrow(user_index)
        recommendingItems = np.asarray(rec.toarray()[0].argsort()[::-1])
        unseen_items_mask = np.in1d(recommendingItems, self.urm[user_index].indices,assume_unique=True, invert=True)
        unseen_items = recommendingItems[unseen_items_mask]
        recommended_items = unseen_items[0:at]
        return recommended_items