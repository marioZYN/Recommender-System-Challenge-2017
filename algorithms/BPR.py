from similarity_cython.SLIM_BPR.Cython.SLIM_BPR_Cython  import SLIM_BPR_Cython
from similarity_cython.CosineSim import Cosine_Similarity
from algorithms.Recommender import Recommender
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import scipy.sparse as sps

class BPR(Recommender):

    def __init__(self):

        super().__init__()
        self.track_track = None
        self.playlist_track = None

    def fit(self,n_iteration, topK, learning_rate, lambda_i, lambda_j):

        #  track-track part

        # print("-- creating attributes")
        # attributes = []
        # attributes.append(self.generate_track_artist_matrix(2))
        # attributes.append(self.generate_track_album_matrix(1))
        # attributes.append(self.generate_track_tag_matrix(1))

        # print("-- combining attributes")
        # result = attributes[0].T
        # for attribute in attributes[1:]:
        #     result = sps.vstack([result.tocoo(), attribute.T.tocoo()])
        #
        # new_urm = sps.vstack([self.urm.tocoo(), result.tocoo()])

        # recommender = SLIM_BPR_Cython(self.urm.tocsr(), recompile_cython=False, positive_threshold=0, sparse_weights=True)
        # recommender.fit(epochs=n_iteration, validate_every_N_epochs=100, URM_test=None, topK=topK,
        #                 batch_size=1, sgd_mode='sgd', learning_rate=learning_rate, lambda_i=lambda_i, lambda_j=lambda_j)
        # self.track_track = recommender.S
        # self.track_track = normalize(self.track_track, norm='l2', axis=1)
        # self.playlist_track = self.urm.tocsr() * self.track_track

        # playlist-playlist part

        recommender = SLIM_BPR_Cython(self.urm.T.tocsr(), recompile_cython=False, positive_threshold=0,                                      sparse_weights=True)
        recommender.fit(epochs=n_iteration, validate_every_N_epochs=100, URM_test=None, topK=topK,
                        batch_size=1, sgd_mode='sgd', learning_rate=learning_rate, lambda_i=lambda_i, lambda_j=lambda_j)
        self.playlist_playlist = recommender.S
        #
        self.playlist_playlist = normalize(self.playlist_playlist, norm='l2', axis=1)

        print("generating prediction matrix")
        self.playlist_track = self.playlist_playlist.tocsr() * self.urm.tocsc()


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

    def generate_track_tag_matrix(self, weight):

        print("* generating track_title_matrix with weight {:.1f}".format(weight))

        # preparing data
        playlist_final = pd.read_csv("../data/tracks_final.csv", sep='\t')
        temp = pd.DataFrame({'track_id': self.tracks_unique})
        track_tag_df = pd.merge(temp, playlist_final[['track_id', 'tags']], on='track_id')

        # calculate index map
        tags = list(set(track_tag_df.tags))
        tags = [x[1:-1].split(', ') for x in tags]
        tags = [item for sublist in tags for item in sublist]
        tag_index_dic = dict(zip(tags, list(np.arange(0, len(tags)))))
        track_index_dic = dict(zip(self.tracks_unique, list(np.arange(0, len(self.tracks_unique)))))

        # unwrap titles
        tags = list(track_tag_df.tags)
        tag_count_list = list(map(lambda x: len(x[1:-1].split(', ')), tags))
        tag_name_list = list(map(lambda x: x[1:-1].split(', '), tags))
        tag_name_list = [item for sublist in tag_name_list for item in sublist] # final use
        track_ids = list(track_tag_df.track_id)
        track_id_list = np.repeat(track_ids, tag_count_list) # final use


        # generate matrix
        ratinglist = [0 if x =='' else weight for x in tag_name_list]
        row_indices = [track_index_dic[x] for x in track_id_list]
        col_indices = [tag_index_dic[x] for x in tag_name_list]
        track_tag_matrix = sps.coo_matrix((ratinglist, (row_indices, col_indices)))
        track_tag_matrix.eliminate_zeros()

        return track_tag_matrix



if __name__ == '__main__':

    from support.utility import read_data
    from support.utility import train_validate_test_split
    import time

    start = time.time()
    print("reading data")
    data = read_data(sample_frac=0.9)
    print("train, test splitting")
    (train, test1, test2) = train_validate_test_split(data)

    print("training ")
    bpr = BPR()
    bpr.setup(train)
    bpr.fit(n_iteration=15, topK=300,learning_rate=1e-3,lambda_i=0.0001,lambda_j=0.0001)
    print("evaluating test1")
    bpr.evaluate_result(train, test1)
    print("evaluating test2")
    bpr.evaluate_result(train, test2)

    print("total time is {:.2f} minutes".format((time.time() - start) / 60))