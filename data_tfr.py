import pickle
import tensorflow as tf
import multiprocessing as mp
import random
import os
from tensorflow.keras.utils import Progbar
from configs import transformed_and_sorted_reviews_path,tfr_dir,count_path,bad_path
from configs import train_fraction, max_length, num_cores,min_length
"""
target negsample : simply sample an item not the target
sequence_negsample : sample items not in sequence for auxiliary loss
"""



"""
transformed_and_sorted_reviews_path = "/root/DIEN/data/tmp/transformed_reviews.pkl"
tfr_dir = "/root/DIEN/data/tfr"
count_path = "/root/DIEN/data/tmp/counts.txt"
bad_path = "/root/DIEN/data/tmp/bad_recs.txt"

train_fraction = 0.95

min_length = 4
max_length = 20
num_cores = 24
"""
def split_data(data,num_cores=num_cores):
    splits = []
    length = len(data)
    per_split = length // num_cores
    for i in range(num_cores-1):
        splits.append(data[i*per_split:(i+1)*per_split])
    splits.append(data[(num_cores-1)*per_split:])
    return splits

def process_single_timeline(timeline,min_length=min_length,max_length=max_length):

    if len(timeline) < min_length+1:
        return None

    

    idx = min_length
    examples = []

    while idx < len(timeline):
        #uID, rating, time, itemID, cate, price
        start = idx-max_length if idx > max_length else 0

        seq = timeline[start:idx]
        target = timeline[idx]

        reviewerID = target[0]
        pos_itemID = target[3]
        pos_cates = target[4]
        pos_price = target[5]
        pos_seq_itemID = [cur[3] for cur in seq]
        pos_seq_price = [cur[5] for cur in seq]
        pos_seq_rating = [cur[1] for cur in seq]
        pos_seq_cates = [cur[4] for cur in seq]

        ctx = {
            "reviewerID" : tf.train.Feature(int64_list = 
                tf.train.Int64List(value = [reviewerID])),

            "pos_itemID" : tf.train.Feature(int64_list = 
                tf.train.Int64List(value = [pos_itemID])),

            "pos_cates" : tf.train.Feature(int64_list = 
                tf.train.Int64List(value = pos_cates)),

            "pos_price" : tf.train.Feature(float_list = 
                tf.train.FloatList(value = [pos_price])),

            "pos_seq_itemID" : tf.train.Feature(int64_list = 
                tf.train.Int64List(value = pos_seq_itemID)),

            "pos_seq_price" : tf.train.Feature(float_list = 
                tf.train.FloatList(value = pos_seq_price)),

            "pos_seq_rating" : tf.train.Feature(float_list = 
                tf.train.FloatList(value = pos_seq_rating)),
        
        }

        ctx = tf.train.Features(feature=ctx)

        seq = {
            "pos_seq_cates" : tf.train.FeatureList(feature = [
                tf.train.Feature(int64_list = tf.train.Int64List(value=cur_cates))
                for cur_cates in pos_seq_cates
            ])
        }
        seq = tf.train.FeatureLists(feature_list = seq)

        example = tf.train.SequenceExample(context=ctx,feature_lists=seq)
        examples.append(example.SerializeToString())

        idx += 1

    return examples

def target_func(split,idx,tfr_dir = tfr_dir, train_fraction=train_fraction,count_path = count_path, bad_path = bad_path):
    length = len(split)
    split = dict(split)
    count = 0
    bad_rec = 0
    
    train_path = os.path.join(tfr_dir,f"train_{idx}.tfrecords")
    test_path = os.path.join(tfr_dir,f"test_{idx}.tfrecords")
    train_writer = tf.io.TFRecordWriter(train_path)
    test_writer = tf.io.TFRecordWriter(test_path)
    prog_bar = Progbar(length)
    for value in split.values():
        timeline = value
        cur_examples = process_single_timeline(timeline)
        if cur_examples is None:
            bad_rec += 1
            continue
        for example in cur_examples:
            rd = random.random()
            if rd > train_fraction:
                test_writer.write(example)
            else:
                train_writer.write(example)
            count += 1

        prog_bar.add(1)

    train_writer.close()
    test_writer.close()

    with open(count_path,'a') as f:
        f.write(f"Process_{idx}::{count}\n")
    with open(bad_path,'a') as f:
        f.write(f"Process_{idx}::{bad_rec}\n")


if __name__ == "__main__":
    #mp.set_start_method('spawn')
    if os.path.exists(transformed_and_sorted_reviews_path):
        with open(transformed_and_sorted_reviews_path,'rb') as f:
            reviews = pickle.load(f)
    else:
        raise RuntimeError("review tmp file does not exist")

    #target_func(reviews)

    reviews = list(reviews.items())
    splits = split_data(reviews)

    ps = []
    for i in range(num_cores):
        p = mp.Process(target=target_func,args=(splits[i],i))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()
    
    print("TFRecords writen!")




    

    

