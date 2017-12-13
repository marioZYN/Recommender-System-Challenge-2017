import pandas as pd
import numpy as np
import scipy.sparse as sps
import time

def similarity_matrix_topk(item_score_matrix, topk=100,verbose=False):

    """
    The function selects the topk most simialer items, column-wise

    :param item_score_matrix:
    :return:
    """

    assert(item_score_matrix.shape[0] == item_score_matrix.shape[1]), "the score matrix is not square"
    start_time = time.time()
    if verbose:
        print("generaing topk matrix")

    # iterate over each column and keep only the topk similar items

    data, row_indices, col_indptr = [], [], []
    item_score_matrix = item_score_matrix.tocsc()
    n_items = item_score_matrix.shape[0]

    for item_index in range(item_score_matrix.shape[0]):
        col_indptr.append(len(data))
        start_position = item_score_matrix.indptr[item_index]
        end_position = item_score_matrix.indptr[item_index+1]
        col_data = item_score_matrix.data[start_position:end_position]
        col_row_index = item_score_matrix.indices[start_position:end_position]

        index_sorted = np.argsort(col_data)
        topk_index = index_sorted[-topk:]

        data.extend(col_data[topk_index])
        row_indices.extend(col_row_index[topk_index])
    col_indptr.append(len(data))

    result = sps.csc_matrix((data, row_indices, col_indptr), shape=(n_items, n_items), dtype=np.float32)
    result = result.tocsr()

    if verbose:
        print("sparse topk matrix generated in {:.2f} seconds".format(time.time()-start_time))

    return result

def compare_results(path1, path2):
    r1 = pd.read_csv(path1)
    r2 = pd.read_csv(path2)
    count = 0
    for i in range(0, 10000):
        tracks1 = set(r2.iloc[i].track_ids.split(" "))
        tracks2 = set(r1.iloc[i].track_ids.split(" "))
        count += len(tracks1 & tracks2)
        print("\r%d playlist completes..." %i,end='', flush=True)

    print()
    print(str(count) + ' similar items')


def precision_score(recommended_items, relevant_items):
    correct = len(set(recommended_items) & set(relevant_items))
    score = correct / float(len(recommended_items))

    return score


def recall_score(recommended_items, relevant_items):
    correct = len(set(recommended_items) & set(relevant_items))
    score = correct / float(len(relevant_items))

    return score


def map_score(recommended_items, relevant_items):
    temp = 0
    for i in range(0, len(recommended_items)):
        temp += precision_score(recommended_items[0:i + 1], relevant_items)
    score = temp / float(len(recommended_items))
    return score


def train_test_split(total, threshold=5):
    temp = total.groupby('playlist_id').filter(lambda x: len(x) > threshold)
    test = temp.groupby('playlist_id').apply(lambda x: x.sample(n=5)).reset_index(drop=True)
    temp = pd.merge(total, test, on=['playlist_id', 'track_id'], indicator=True, how='outer')
    train = temp[temp['_merge'] != 'both'][['playlist_id', 'track_id']]

    return train, test


def train_validate_test_split(total, threshold=10, select_number=10):

    temp = total.groupby('playlist_id').filter(lambda x: len(x) > threshold)
    test = temp.groupby('playlist_id').apply(lambda x: x.sample(n=select_number)).reset_index(drop=True)
    temp = pd.merge(total, test, on=['playlist_id', 'track_id'], indicator=True, how='outer')
    train = temp[temp['_merge'] != 'both'][['playlist_id', 'track_id']]
    (validate, test) = train_test_split(test)
    return train, validate, test


def read_data(sample_frac=1, only_target=False):

    if only_target:
        # train_final = pd.read_csv("../data/train_final.csv", sep='\t')
        # target_tracks = pd.read_csv("../data/target_tracks.csv")
        # data = pd.merge(train_final, target_tracks, on='track_id')
        data = pd.read_csv("../data/train_final_pruned_tracks.csv", sep='\t')
        return data.sample(frac=sample_frac)
    else :
        train_final = pd.read_csv("../data/train_final.csv", sep='\t')
        return train_final.sample(frac=sample_frac)
