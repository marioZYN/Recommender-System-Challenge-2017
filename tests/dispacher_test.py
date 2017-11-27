from algorithms.BPR import BFR
from algorithms.CollaborativeFilterItem import  CollaborativeFilterItem
from algorithms.Dispacher import  Dispacher
from support import read_data
from support import train_test_split
from support import train_validate_test_split
import time

start = time.time()
print("reading data...")
data = read_data(sample_frac=0.9)
print("train, validate, test splitting...")
(train, validate, test) = train_validate_test_split(data)

print("training BFR system ")
bfr = BFR()
bfr.setup(train)
bfr.fit(n_iteration=20, topK=200, learning_rate=1e-3, lambda_i=0.0001, lambda_j=0.0001, combine_weight=[1,1], artist_weight=1, album_weight=0.3, artist_cfu_weight=0)
bfr.evaluate_result(train, test)

print("training cfi rec system ")
cfi = CollaborativeFilterItem()
cfi.setup(train)
cfi.fit(artist_weight=1, album_weight=0.3)
cfi.evaluate_result(train, test)

print("dispacher evaluating...")
dispacher = Dispacher()
dispacher.evaluate(train=train, validate=validate, test=test, rec1=bfr, rec2=cfi)

print("total time is {:.2f} minutes".format((time.time()-start)/60))