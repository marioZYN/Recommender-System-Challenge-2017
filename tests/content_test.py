from algorithms.ContentBased import ContentBased
from utility import read_data
from utility import train_test_split
import time

start = time.time()
print("reading data...")
data = read_data(sample_frac=0.2)
print("train, test splitting...")
(train, test) = train_test_split(data, 9)

print("training content-based rec system ")
cb = ContentBased()
cb.setup(train)
cb.fit(tag_flag=True, tag_weight=1, artist_flag=True, artist_weight=50, album_flag=True, album_weight=10)
cb.evaluate_result(train, test)


print("total time is {:.2f} minutes".format((time.time()-start)/60))