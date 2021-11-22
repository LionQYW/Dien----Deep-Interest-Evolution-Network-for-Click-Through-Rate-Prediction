import pickle
import os

#a dir for tmp file, e.g. will not be used when train the model
tmp_dir = "/root/DIEN/data/tmp"
#see the filename and you will understand its meaning
raw_review_path = "/root/DIEN/data/Books_5.json"
raw_meta_data_path = "/root/DIEN/data/meta_Books.json"
#a file that contain size of sparse features for model embedding layer
n_sparse_path = "/root/DIEN/data/tmp/n_sparse.txt"

RESERVE_FOR_PAD = "RESERVE_FOR_PAD"
#multi processing
num_cores = 24
# 24 too large for final step, mp copy all resources for each process
another_num_cores = 12
# hush buckets number for missing value, a missing value is randomly placed into a buckets
num_hash_buckets = 100

#see the filename, a file contains filtered and transformed data
transformed_and_sorted_reviews_path = "/root/DIEN/data/tmp/transformed_reviews.pkl"
#dir for tfrecord files
tfr_dir = "/root/DIEN/data/tfr"
#not important, count how many bad entries in dataset, very small, aound 1k from 22m
count_path = "/root/DIEN/data/tmp/counts.txt"
bad_path = "/root/DIEN/data/tmp/bad_recs.txt"
#0.95 for training, 0.05 for test
train_fraction = 0.95
#min and max timestamp length
min_length = 4
max_length = 20

#Got sparse feature num for model embedding and negtive sampling
with open(n_sparse_path,'r') as f:
    nums = f.readlines()
n_reviewer = int(nums[0].strip('\n').split('::')[1])
n_cates = int(nums[1].strip('\n').split("::")[1])
n_item = int(nums[2].strip('\n').split("::")[1])


train_global_batch_size = 20480
test_global_batch_size = 2048

with open(os.path.join(tfr_dir,"item_feat_list.pkl"),'rb') as f:
    cates_list = pickle.load(f)
    price_list = pickle.load(f)

epochs = 20

################################################################
#should uncomment below codes first to get these two list above!
################################################################
"""
#itemid -> itemid, cates, price

item_map_file = "/root/DIEN/data/tmp/item_feat_map.pkl"
with open(item_map_file,'rb') as f:
    item_map = pickle.load(f)

cates_list = [[0,0,0]]
price_list = [0.0]

for id in range(1,n_item):
    cates_list.append(item_map[id][1])
    price_list.append(item_map[id][2])

with open(os.path.join(tfr_dir,"item_feat_list.pkl"),'wb') as f:
    pickle.dump(cates_list,f)
    pickle.dump(price_list,f)
"""