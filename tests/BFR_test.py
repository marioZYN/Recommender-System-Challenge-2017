from algorithms.BPR import BFR
from utility import read_data
from utility import train_test_split
import time

start = time.time()
print("reading data...")
data = read_data(sample_frac=0.9)
print("train, test splitting...")
(train, test) = train_test_split(data, 5)

print("training content-based rec system ")
bfr = BFR()
bfr.setup(train)
bfr.fit(n_iteration=15, topK=300, learning_rate=1e-3, lambda_i=0.0001, lambda_j=0.0001, combine_weight=[1,1], artist_weight=0, album_weight=0, artist_cfu_weight=1, vstack_weight=1)
bfr.evaluate_result(train, test)


print("total time is {:.2f} minutes".format((time.time()-start)/60))