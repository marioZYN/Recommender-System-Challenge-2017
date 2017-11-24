from algorithms.CollaborativeFilterItem import CollaborativeFilterItem
from algorithms.CollaborativeFilterUser import CollaborativeFilterUser
from algorithms.ContentBased import ContentBased
from algorithms.Hybrid import Hybrid
from utility import read_data
from utility import train_test_split
import time

start = time.time()
print("reading data...")
data = read_data(sample_frac=0.8)
print("train, test splitting...")
(train, test) = train_test_split(data, 9)
# train = data

print("training cfi...")
cfi = CollaborativeFilterItem()
cfi.setup(train)
cfi.fit()
cfi.evaluate_result(train, test)

print("training cfu...")
cfu = CollaborativeFilterUser()
cfu.setup(train)
cfu.fit()
cfu.evaluate_result(train, test)

print("training cb...")
cb = ContentBased()
cb.setup(train)
cb.fit(artist_flag=True, artist_weight=4, album_flag=True, album_weight=1, tag_flag=False, tag_weight=0.2)
cb.evaluate_result(train, test)



print("combining...")
hybrid = Hybrid()
hybrid.setup(train)
hybrid.fit(recommender_objects=[cfi,cfu,cb],weights=[1,1,1])
print("evaluating...")
hybrid.evaluate_result(train, test)

hybrid.fit(recommender_objects=[cfi,cfu,cb],weights=[1,0.1,0.5])
print("evaluating...")
hybrid.evaluate_result(train, test)

hybrid.fit(recommender_objects=[cfi,cfu,cb],weights=[1,0.2,0.6])
print("evaluating...")
hybrid.evaluate_result(train, test)
# cfi.gen_result("./Results/hybrid_new.csv")

print("total time is {:.2f} minutes".format((time.time()-start)/60))