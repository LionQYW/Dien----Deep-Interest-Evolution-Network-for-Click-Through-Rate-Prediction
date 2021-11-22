import tensorflow as tf
import random 
import numpy as np
from collections import defaultdict
import re
import json
import pickle
import os
from multiprocessing import Process, Manager
import multiprocessing as mp
import gc
import copy
from configs import tmp_dir, raw_review_path, raw_meta_data_path, n_sparse_path
from configs import RESERVE_FOR_PAD, num_cores, another_num_cores, num_hash_buckets

"""
#Preserve original global variable defination, for debug purpose
#to place tmp files, not final data
tmp_dir = "/root/DIEN/data/tmp"
raw_review_path = "/root/DIEN/data/Books_5.json"
raw_meta_data_path = "/root/DIEN/data/meta_Books.json"
n_sparse_path = "/root/DIEN/data/tmp/n_sparse.txt"

RESERVE_FOR_PAD = "RESERVE_FOR_PAD"


num_cores = 24
another_num_cores = 12 #24 too large for final step, mp copy all resources for each process

num_hash_buckets = 100
"""

missing_hash_buckets = [f"hash_buckets_for_missing_value{i}" for i in range(num_hash_buckets)]

def split_data(data,num_cores=num_cores):
    splits = []
    length = len(data)
    per_split = length // num_cores
    for i in range(num_cores-1):
        splits.append(data[i*per_split:(i+1)*per_split])
    splits.append(data[(num_cores-1)*per_split:])
    return splits

def extract_price(price):
    sc = re.search("\d+\.\d+",price)
    if sc:
        num = float(sc.group())
    else:
        num = 0.0
    return np.log(1+num)

def get_all_items(raw_meta_data_path=raw_meta_data_path):
    with open(raw_meta_data_path,"rb") as f:
        data = f.readlines()

    splits = split_data(data)

    def _target_func(split, mplist):
        tmp_dict = {}
        for record in split:
            record = eval(record)
            asin = record['asin']
            category = record['category'] if record['category'] else [
                    missing_hash_buckets[random.randint(0,num_hash_buckets-1)]]
            price = extract_price(record['price'])

            tmp_dict[asin] = [asin,category,price]
        mplist.append(tmp_dict)

    with Manager() as manager:
        ps = []
        mplist = manager.list()
        for i in range(num_cores):        
            p = Process(target=_target_func,args=(splits[i],mplist))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()

        item_dict = {}
        for dic in mplist:
            item_dict.update(dic)
        return item_dict

def get_all_reviews(item_dict,raw_review_path=raw_review_path):
    with open(raw_review_path,"rb") as f:
        data = f.readlines()

    splits = split_data(data)

    def _target_func(split,mplist,bad_records):
        user_reviews = defaultdict(list)
        bad_count = 0
        for record in split:
            try:
                record = json.loads(record)
                reviewerID = record['reviewerID']
                asin = record['asin']
                overall = record['overall'] / 5.0
                timestamp = record['unixReviewTime']
                assert type(timestamp) == int

                cur_entry = [reviewerID, overall, timestamp] + item_dict[asin]
                user_reviews[reviewerID].append(cur_entry)
            except:
                bad_count += 1
        #ID, rating, time, itemID, cate, price
        mplist.append(user_reviews)
        bad_records.append(bad_count)

    with Manager() as manager:
        ps = []
        mplist = manager.list()
        bad_counts = manager.list()
        for i in range(num_cores):
            p = Process(target=_target_func,args=(splits[i],mplist,bad_counts))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()

        user_reviews = defaultdict(list)
        for dic in mplist:
            for reviewerID in dic.keys():
                user_reviews[reviewerID] += dic[reviewerID]
        return user_reviews, sum(bad_counts)

def get_vocabs(user_reviews,n_sparse_path=n_sparse_path):
    user_reviews = list(user_reviews.items())
    splits = split_data(user_reviews)

    def _target_func(split,ulist, clist, ilist):
        split = dict(split)
        userID, cates, itemID = set(), set(), set()
        for user, user_reviews in split.items():
            userID.add(user)

            for review in user_reviews:
                _, _, _, item_ID_, categories_, _ = review
                cates = cates | set(categories_)
                itemID.add(item_ID_)
        ulist.append(userID)
        clist.append(cates)
        ilist.append(itemID)

    with Manager() as manager:
        ps = []
        ulist, clist, ilist = manager.list(), manager.list(), manager.list()
        for i in range(num_cores):
            p = Process(target=_target_func,args=(splits[i],ulist,clist,ilist))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()

        userID, cates, itemID = set(), set(), set()
        for (u,c,i) in zip(ulist,clist,ilist):
            userID = userID | u
            cates = cates | c
            itemID = itemID | i

        userID = [RESERVE_FOR_PAD] + list(userID)
        n_user = len(userID)
        userID = {userID[i]:i for i in range(len(userID))}

        cates = [RESERVE_FOR_PAD] + list(cates)
        n_cates = len(cates)
        cates = {cates[i]:i for i in range(len(cates))}

        itemID = [RESERVE_FOR_PAD] + list(itemID)
        n_item = len(itemID)
        itemID = {itemID[i]:i for i in range(len(itemID))}

        with open(n_sparse_path,'w') as f:
            f.write("user"+"::"+str(n_user)+"\n")
            f.write("cates"+"::"+str(n_cates)+"\n")
            f.write("item"+"::"+str(n_item)+"\n")

        return userID, cates, itemID


def sort_and_transform(userID, cates, itemID, user_reviews):
    user_reviews = list(user_reviews.items())
    splits = split_data(user_reviews,num_cores=another_num_cores)

    def _target_func(split,mplist):
        split = dict(split)
        for user in split.keys():
            split[user].sort(key = lambda x:x[2])
            
            for i in range(len(split[user])):
                user_id_, _, _, item_id_, categories_, _ = split[user][i]
                user_id_ = userID[user_id_]
                item_id_ = itemID[item_id_]
                categories_ = [cates[cur] for cur in categories_]

                split[user][i][0] = user_id_
                split[user][i][3] = item_id_
                split[user][i][4] = categories_
        mplist.append(split)

    with Manager() as manager:
        ps = []
        mplist = manager.list()
        user_reviews = defaultdict(list)
        for i in range(another_num_cores):
            p = Process(target=_target_func,args=(splits[i],mplist))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()

        for dic in mplist:
            for user in dic.keys():
                user_reviews[user] += dic[user]

        return user_reviews

def build_item_feat_map(item_dict_, itemID, cates):
    item_dict = {}
    bad_count = 0
    for key,v in item_dict_.items():
        try:
            item_dict[itemID[key]] = [itemID[v[0]],[cates[cur] for cur in v[1]],v[2]]
        except:
            bad_count += 1
    print(bad_count)
    print(len(item_dict.items()))
    return item_dict


if __name__ == "__main__":
    #mp.set_start_method('forkserver') 
    item_dict_path = os.path.join(tmp_dir,"item_dict.pkl")
    if not os.path.exists(item_dict_path):
        item_dict = get_all_items()
        with open(item_dict_path,'wb') as f:
            pickle.dump(item_dict,f)
    else:
        with open(item_dict_path,'rb') as f:
            item_dict = pickle.load(f)
    gc.collect()
    
    user_reviews_path = os.path.join(tmp_dir,"user_reviews.pkl")
    if not os.path.exists(user_reviews_path):
        user_reviews,bad_counts = get_all_reviews(item_dict)
        print(f"there all {bad_counts} records can not find raw data info in reviews")
        with open(user_reviews_path,'wb') as f:
            pickle.dump(user_reviews,f)
    else:
        with open(user_reviews_path,'rb') as f:
            user_reviews = pickle.load(f)
    gc.collect()
    #print(user_reviews)
    
    vocabs_path = os.path.join(tmp_dir,"vocabs.pkl")
    if not os.path.exists(vocabs_path):
        userID, cates, itemID = get_vocabs(user_reviews)
        with open(vocabs_path,'wb') as f:
            pickle.dump((userID,cates,itemID),f)
    else:
        with open(vocabs_path,'rb') as f:
            (userID,cates,itemID) = pickle.load(f)
    gc.collect()
    #print(userID)
    #print(cates)
    #print(itemID)
    item_feat_map_path = os.path.join(tmp_dir,"item_feat_map.pkl")
    if not os.path.exists(item_feat_map_path):
        item_feat_map = build_item_feat_map(item_dict_=item_dict,itemID=itemID,cates=cates)
        with open(item_feat_map_path,'wb') as f:
            pickle.dump(item_feat_map,f)

    trans_revs_path = os.path.join(tmp_dir,"transformed_reviews.pkl")
    if not os.path.exists(trans_revs_path):
        trans_revs = sort_and_transform(userID=userID,cates=cates,itemID=itemID,user_reviews=user_reviews)
        with open(trans_revs_path,'wb') as f:
            pickle.dump(trans_revs,f)
    else:
        with open(trans_revs_path,'rb') as f:
            trans_revs = pickle.load(f)
    gc.collect()

