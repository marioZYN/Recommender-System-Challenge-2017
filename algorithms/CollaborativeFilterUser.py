from algorithms.Recommender import Recommender
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import scipy.sparse as sps

class CollaborativeFilterUser(Recommender):


    def __init__(self):

        super().__init__()
        self.playlist_playlist = None
        self.playlist_track = None

    def fit(self):

        print("-- calculating user-user similarity matrix")
        playlist_playlist = cosine_similarity(self.urm, dense_output=False)
        print("-- calculating user-user similarity from titles")
        playlist_title_matrix = self.generate_playlist_title_matrix(1)
        playlist_playlist_titles = cosine_similarity(playlist_title_matrix, dense_output=False)
        print("-- adding cfu and content based")
        self.playlist_playlist = playlist_playlist - playlist_playlist_titles * 0.4
        print("-- generating prediction matrix")
        self.playlist_track = self.playlist_playlist.tocsr() * self.urm.tocsc()

    def generate_playlist_owner_matrix(self, weight):

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

        return playlist_owner_matrix

    def generate_playlist_title_matrix(self, weight):

        print("* generating playlist_title_matrix with weight {:.1f}".format(weight))

        # preparing data
        playlist_final = pd.read_csv("../data/playlists_final.csv", sep='\t')
        temp = pd.DataFrame({'playlist_id': self.playlist_unique})
        playlist_title_df = pd.merge(temp, playlist_final[['playlist_id', 'title']], on='playlist_id')

        # calculate index map
        titles = list(set(playlist_title_df.title))
        titles = [x[1:-1].split(', ') for x in titles]
        titles = [item for sublist in titles for item in sublist]
        title_index_dic = dict(zip(titles, list(np.arange(0, len(titles)))))
        playlist_index_dic = dict(zip(self.playlist_unique, list(np.arange(0, len(self.playlist_unique)))))

        # unwrap titles
        titles = list(playlist_title_df.title)
        title_count_list = list(map(lambda x: len(x[1:-1].split(', ')), titles))
        title_name_list = list(map(lambda x: x[1:-1].split(', '), titles))
        title_name_list = [item for sublist in title_name_list for item in sublist] # final use
        playlist_ids = list(playlist_title_df.playlist_id)
        playlist_id_list = np.repeat(playlist_ids, title_count_list) # final use


        # generate matrix
        ratinglist = [0 if x =='' else weight for x in title_name_list]
        row_indices = [playlist_index_dic[x] for x in playlist_id_list]
        col_indices = [title_index_dic[x] for x in title_name_list]
        playlist_title_matrix = sps.coo_matrix((ratinglist, (row_indices, col_indices)))
        playlist_title_matrix.eliminate_zeros()

        return playlist_title_matrix

    def add_artist(self, weight):

        tracks_final = pd.read_csv("./data/tracks_final.csv", sep='\t')
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
        tracks_final = pd.read_csv("./data/tracks_final.csv", sep='\t')
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
        rec = self.playlist_track.getrow(user_index)
        recommendingItems = np.asarray(rec.toarray()[0].argsort()[::-1])
        unseen_items_mask = np.in1d(recommendingItems, self.urm[user_index].indices, assume_unique=True, invert=True)
        unseen_items = recommendingItems[unseen_items_mask]
        recommended_items = unseen_items[0:at]
        return recommended_items


if __name__ == '__main__':

    from support.utility import read_data
    from support.utility import train_test_split
    import time

    start = time.time()
    print("reading data")
    data = read_data(sample_frac=0.9)
    print("train, test splitting")
    (train, test) = train_test_split(data, 5)

    print("training cfu")
    cfi = CollaborativeFilterUser()
    cfi.setup(train)
    cfi.fit()
    print("evaluating")
    cfi.evaluate_result(train, test)

    print("total time is {:.2f} minutes".format((time.time() - start) / 60))
