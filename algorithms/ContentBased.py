from algorithms.Recommender import Recommender
from sklearn.metrics.pairwise import cosine_similarity
from similarity_cython.CosineSim import Cosine_Similarity
from sklearn.preprocessing import normalize
import pandas as pd
import scipy.sparse as sps
import numpy as np

class ContentBased(Recommender):

    def __init__(self):

        super().__init__()
        self.track_track = None
        self.playlist_track = None

    def fit(self, artist=0, album=0, tag=0, owner=0, title=0, history =0):

        print("-- creating attributes")
        attributes = []
        if artist != 0:
            attributes.append(self.generate_track_artist_matrix(artist))
        if album != 0:
            attributes.append(self.generate_track_album_matrix(album))
        if tag != 0:
            attributes.append(self.generate_track_tag_matrix(tag))
        if owner != 0:
            attributes.append(self.generate_track_owner_matrix(owner))
        if title != 0:
            attributes.append(self.generate_track_title_matrix(title))
        if history != 0:
            attributes.append(self.urm.T * history)

        print("-- combining attributes")
        result = attributes[0]
        for attribute in attributes[1:]:
            result = sps.hstack([result.tocoo(), attribute.tocoo()])

        print("-- generating cosine similarity matrix")
        cos = Cosine_Similarity(result.T.tocsr(), TopK=2000)
        self.track_track = cos.compute_similarity()
        self.track_track = normalize(self.track_track, norm='l2', axis=1)

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

    def generate_track_tag_matrix(self, weight):

        print("* generating track_tag_matrix with weight {:.1f}".format(weight))

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

    def generate_track_title_matrix(self, weight):

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
        track_title_matrix = self.urm.T.tocsr() * playlist_title_matrix

        return track_title_matrix



    # def recommend(self, user_id, at=5):
    #
    #     user_index = self.playlist_dic[user_id]
    #     rec = self.playlist_track.tocsr().getrow(user_index)
    #     recommendingItems = np.asarray(rec.toarray()[0].argsort()[::-1])
    #     unseen_items_mask = np.in1d(recommendingItems, self.urm[user_index].indices,assume_unique=True, invert=True)
    #     unseen_items = recommendingItems[unseen_items_mask]
    #     recommended_items = unseen_items[0:at]
    #     return recommended_items


if __name__ == "__main__":

    from support.utility import read_data
    from support.utility import train_test_split
    from support.utility import train_validate_test_split
    import time

    start = time.time()
    print("reading data")
    data = read_data(sample_frac=0.9, only_target=True)
    print("train, test splitting")
    (train, test1, test2) = train_validate_test_split(data)

    print("training cb item")
    cb_item = ContentBased()
    cb_item.setup(train)
    cb_item.fit(artist=15, album=15, owner=2, tag=1, history=5)

    # cb_item.gen_result('../results/cb_item_v6.csv')
    print("evaluating test1")
    cb_item.evaluate_result(train, test1)

    print("evaluating test2")
    cb_item.evaluate_result(train, test2)

    print("total time is {:.2f} minutes".format((time.time() - start) / 60))