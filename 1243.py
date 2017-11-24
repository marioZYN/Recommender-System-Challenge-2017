from algorithms.BPR import BFR
from algorithms.CollaborativeFilterItem import  CollaborativeFilterItem
from algorithms.Hybrid import Hybrid
from utility import read_data
import pandas as pd
from utility import train_test_split
import time

start = time.time()
print("reading data...")
train_final = pd.read_csv("./Data/train_final.csv", sep='\t')
train = train_final

print("training BFR system ")
bfr = BFR()
bfr.setup(train)
bfr.fit(n_iteration=20, topK=300, learning_rate=1e-3, lambda_i=0.0001, lambda_j=0.0001, combine_weight=[1,1], artist_weight=1, album_weight=0.3, artist_cfu_weight=0)

print("training cfi rec system ")
cfi = CollaborativeFilterItem()
cfi.setup(train)
cfi.fit(combine_weight=0.2, artist=1, album=1,topk=3000)

print("combining...")
hybrid = Hybrid()
hybrid.setup(train)
hybrid.fit(recommender_objects=[cfi,bfr],weights=[1,1])
print("generating...")
hybrid.gen_result("./Results/hybrid_v3.csv")

print("total time is {:.2f} minutes".format((time.time()-start)/60))