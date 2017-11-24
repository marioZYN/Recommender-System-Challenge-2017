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

print("training ")
cfi = CollaborativeFilterItem()
cfi.setup(train)
cfi.fit(combine_weight=0.4, artist=1, album=1,topk=3000)

cfi.gen_result("./Results/cfi_0903.csv")

print("total time is {:.2f} minutes".format((time.time()-start)/60))