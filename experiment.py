from algorithms.BPR import BFR
from algorithms.CollaborativeFilterItem import CollaborativeFilterItem
from utility import read_data
from utility import train_test_split
import time

start = time.time()
print("reading data...")
data = read_data(sample_frac=0.9)
print("train, test splitting...")
(train, test) = train_test_split(data, 5)

cfi = CollaborativeFilterItem()
cfi.setup(train)
cfi.fit(combine_weight=0.2, artist=1, album=1,topk=3000)
print("combine_weight=0.2, artist=1, album=1,topk=3000")
cfi.evaluate_result(train, test)

cfi = CollaborativeFilterItem()
cfi.setup(train)
cfi.fit(combine_weight=0.2, artist=1, album=1,topk=4000)
print("combine_weight=0.2, artist=1, album=1,topk=4000")
cfi.evaluate_result(train, test)

cfi = CollaborativeFilterItem()
cfi.setup(train)
cfi.fit(combine_weight=0.2, artist=1, album=1,topk=5000)
print("combine_weight=0.2, artist=1, album=1,topk=5000")
cfi.evaluate_result(train, test)

cfi = CollaborativeFilterItem()
cfi.setup(train)
cfi.fit(combine_weight=0.1, artist=1, album=1,topk=3000)
print("combine_weight=0.1, artist=1, album=1,topk=3000")
cfi.evaluate_result(train, test)

cfi = CollaborativeFilterItem()
cfi.setup(train)
cfi.fit(combine_weight=0.2, artist=1, album=2,topk=3000)
print("combine_weight=0.2, artist=1, album=2,topk=3000")
cfi.evaluate_result(train, test)

cfi = CollaborativeFilterItem()
cfi.setup(train)
cfi.fit(combine_weight=2, artist=1, album=1,topk=3000)
print("combine_weight=2, artist=1, album=1,topk=3000")
cfi.evaluate_result(train, test)

print("total time is {:.2f} minutes".format((time.time()-start)/60))