from algorithms.CollaborativeFilterUser import CollaborativeFilterUser
from support import read_data
from support import train_test_split
import time

start = time.time()
print("reading data")
data = read_data(sample_frac=0.9)
print("train, test splitting")
(train, test) = train_test_split(data, 5)

print("training cfu")
cfu = CollaborativeFilterUser()
cfu.setup(train)
cfu.fit()
cfu.evaluate_result(train, test)


print("total time is {:.2f} minutes".format((time.time()-start)/60))