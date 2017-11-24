import pandas as pd


def compare_results(path1, path2):
    r1 = pd.read_csv(path1)
    r2 = pd.read_csv(path2)
    count = 0
    for i in range(0, 10000):
        tracks1 = set(r2.iloc[i].track_ids.split(" "))
        tracks2 = set(r1.iloc[i].track_ids.split(" "))
        count += len(tracks1 & tracks2)
        print("\r%d playlist completes..." % i, end='', flush=True)

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

def read_data(sample_frac=1):
    train_final = pd.read_csv("./Data/train_final.csv", sep='\t')
    return train_final.sample(frac=sample_frac)