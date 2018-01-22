from algorithms.Recommender import Recommender
from similarity_cython.CosineSim import Cosine_Similarity
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import scipy.sparse as sps
from support.utility import similarity_matrix_topk

class CollaborativeFilterUser(Recommender):


    def __init__(self):

        super().__init__()
        self.playlist_playlist = None
        self.playlist_track = None

    def fit(self):

        cos = Cosine_Similarity(self.urm.T, TopK=2000)
        self.playlist_playlist = cos.compute_similarity()
        self.playlist_playlist = normalize(self.playlist_playlist, norm='l2', axis=1)
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

    def generate_playlist_artist_matrix(self, weight):

        print("* generating playlist_artist_matrix with weight {:.1f}".format(weight))

        # reading data and perform prune

        tracks_final = pd.read_csv("../data/tracks_final.csv", sep='\t')
        temp = pd.DataFrame({'track_id': self.tracks_unique})
        track_artist_df = pd.merge(temp, tracks_final[['track_id', 'artist_id']], on='track_id')
        artists = list(set(track_artist_df.artist_id))
        track_id_artist_id_dic = dict(zip(list(track_artist_df.track_id), list(track_artist_df.artist_id)))
        artist_id_index_dic = dict(zip(artists, list(np.arange(0, len(artists)))))

        # generating track_aritst_matrix

        ratinglist = [weight] * len(self.tracks_unique)
        row_indices = [x for x in range(0, len(self.tracks_unique))]
        col_indices = list(map(lambda x: artist_id_index_dic[track_id_artist_id_dic[x]], self.tracks_unique))
        track_artist_matrix = sps.coo_matrix((ratinglist, (row_indices, col_indices)))

        # generating playlist_artist_matrix

        playlist_artist_matrix = self.urm.tocsr() * track_artist_matrix.tocsc()

        return playlist_artist_matrix

    def generate_playlist_album_matrix(self, weight):

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

        # generating playlist_album_matrix

        playlist_album_matrix = self.urm.tocsr() * track_album_matrix.tocsc()

        return playlist_album_matrix



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
