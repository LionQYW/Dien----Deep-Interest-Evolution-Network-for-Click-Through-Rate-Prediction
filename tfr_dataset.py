import tensorflow as tf
from configs import n_item, num_cores, tfr_dir,cates_list, price_list #item_map, 
from configs import train_global_batch_size, test_global_batch_size
import random
import os

ctx_dict = {
    "reviewerID" : tf.io.FixedLenFeature((),tf.int64),
    "pos_itemID" : tf.io.FixedLenFeature((),tf.int64),
    "pos_cates" : tf.io.VarLenFeature(tf.int64),
    "pos_price" : tf.io.FixedLenFeature((),tf.float32),
    "pos_seq_itemID" : tf.io.VarLenFeature(tf.int64),
    "pos_seq_price" : tf.io.VarLenFeature(tf.float32),
    "pos_seq_rating" : tf.io.VarLenFeature(tf.float32),
}

seq_dict = {
    "pos_seq_cates" : tf.io.VarLenFeature(tf.int64)
}

#[n_item,ragged]
cates_list = tf.ragged.constant(cates_list,dtype=tf.int64)
#[n_item,cates]
cates_list = cates_list.to_tensor()
#[n_item]
price_list = tf.convert_to_tensor(price_list, dtype=tf.float32)

def batch_parse(x):
    parsed = tf.io.parse_sequence_example(x,context_features=ctx_dict,sequence_features=seq_dict)
    feats = {}
    for dic in parsed:
        feats.update(dic)
    for key, value in feats.items():
        if type(value) == tf.sparse.SparseTensor:
            feats[key] = tf.sparse.to_dense(value)
    
    pos_seq_items = feats['pos_seq_itemID']
    #Batch * Length
    seq_shape = tf.shape(pos_seq_items)
    num_item_in_pos_seq = seq_shape[0] * seq_shape[1]
    pos_seq_items = tf.reshape(pos_seq_items,shape=(-1,))

    candidates = tf.random.uniform(shape=(num_item_in_pos_seq,),minval=1,maxval=n_item,dtype=tf.int64)
    

    random_shift = tf.random.uniform(shape=(num_item_in_pos_seq,),minval=-1000,maxval=1000,dtype=tf.int64)
    random_shift = pos_seq_items + random_shift
    #larger than 0, 0 is pad
    not_less = tf.cast(tf.greater(random_shift,0),tf.int64)
    #smaller than n_item
    not_greater = tf.cast(tf.less(random_shift,n_item),tf.int64)
    #meet both requirements
    legal_shift = tf.greater((not_less + not_greater),1)
    #if not meet, replace it with candidates
    random_shift = tf.where(legal_shift,random_shift,candidates)
    #candidates as same as pos_seq_items is invalid
    invalid_candidates = tf.equal(pos_seq_items,candidates)
    #if it is invalid, replace candidates with randomshift
    neg_seq_id = tf.where(invalid_candidates,random_shift,candidates)

    neg_seq_cates = tf.gather(cates_list,indices=neg_seq_id,axis=0)
    neg_seq_price = tf.gather(price_list,indices=neg_seq_id,axis=0)
    ratings = tf.constant([0.2,0.4,0.6,0.8,1.0],dtype=tf.float32)
    random_rating_indices = tf.random.uniform(shape = (num_item_in_pos_seq,),minval=0,maxval=5,dtype=tf.int64)
    neg_seq_rating = tf.gather(ratings,indices=random_rating_indices)

    neg_seq_id = tf.reshape(neg_seq_id,shape=seq_shape)
    neg_seq_cates = tf.reshape(neg_seq_cates,shape=(seq_shape[0],seq_shape[1],-1))
    neg_seq_price = tf.reshape(neg_seq_price,shape=seq_shape)
    neg_seq_rating = tf.reshape(neg_seq_rating,shape=seq_shape)
    

    pos_items = feats['pos_itemID']
    target_length = tf.shape(pos_items)
    candidates = tf.random.uniform(shape=(target_length),minval=1,maxval=n_item,dtype=tf.int64)

    random_shift = tf.random.uniform(shape=(target_length),minval=-1000,maxval=1000,dtype=tf.int64)
    random_shift = pos_items + random_shift
    #larger than 0, 0 is pad
    not_less = tf.cast(tf.greater(random_shift,0),tf.int64)
    #smaller than n_item
    not_greater = tf.cast(tf.less(random_shift,n_item),tf.int64)
    #meet both requirements
    legal_shift = tf.greater((not_less + not_greater),1)
    #if not meet, replace it with candidates
    random_shift = tf.where(legal_shift,random_shift,candidates)
    #candidates as same as pos_seq_items is invalid
    invalid_candidates = tf.equal(pos_items,candidates)
    #if it is invalid, replace candidates with randomshift
    neg_target = tf.where(invalid_candidates,random_shift,candidates)
    neg_price = tf.gather(price_list,indices=neg_target,axis=0)
    neg_cates = tf.gather(cates_list,indices=neg_target,axis=0)

    return (feats['reviewerID'],
    feats['pos_seq_itemID'],feats['pos_seq_cates'],feats['pos_seq_price'],feats['pos_seq_rating'],
    neg_seq_id,neg_seq_cates,neg_seq_price,neg_seq_rating,
    feats['pos_itemID'],feats['pos_cates'],feats['pos_price'],
    neg_target,neg_cates,neg_price)

#tune the input data structure
def batch_tune(reviwerID, pos_seq_itemID, pos_seq_cates, pos_seq_price, pos_seq_rating, \
        neg_seq_itemID, neg_seq_cates, neg_seq_price,neg_seq_rating, \
        pos_itemID, pos_cates, pos_price, \
        neg_itemID, neg_cates, neg_price):

    output = {}
    output['reviewerID'] = reviwerID
    output['pos_seq'] = (pos_seq_itemID, pos_seq_cates, pos_seq_price, pos_seq_rating)
    output['neg_seq'] = (neg_seq_itemID, neg_seq_cates, neg_seq_price,neg_seq_rating)
    output['pos_target'] = (pos_itemID, pos_cates, pos_price)
    output['neg_target'] = (neg_itemID, neg_cates, neg_price)

    return output


train_files = [os.path.join(tfr_dir,f"train_{i}.tfrecords") for i in range(num_cores)]
train_ds = tf.data.TFRecordDataset(train_files,num_parallel_reads=num_cores)
train_ds = train_ds.cache().shuffle(20000,reshuffle_each_iteration=True).batch(
    train_global_batch_size,drop_remainder=True).map(batch_parse).map(batch_tune)

test_files = [os.path.join(tfr_dir,f"test_{i}.tfrecords") for i in range(num_cores)]
test_ds = tf.data.TFRecordDataset(test_files,num_parallel_reads=num_cores)
test_ds = test_ds.batch(test_global_batch_size,drop_remainder=True).map(batch_parse).map(batch_tune)













#Above is optimized random negsampling, 0.48s/per batch (10240)
#Below is naive python loop negsampling, 5s/per batch (1024)
"""
def train_parse(x):
    parsed = tf.io.parse_single_sequence_example(x,context_features=ctx_dict,sequence_features=seq_dict)
    feats = {}
    for dic in parsed:
        feats.update(dic)
    for key, value in feats.items():
        if type(value) == tf.sparse.SparseTensor:
            feats[key] = tf.sparse.to_dense(value)
    
    #Negtive Sampling for historical rating sequence
    pos_seq_items = feats['pos_seq_itemID'].numpy()

    neg_seq_id = []
    neg_seq_price = []
    neg_seq_cates = []
    neg_seq_rating = []


    m = 0
    rates = [0.2,0.4,0.6,0.8,1.0]
    while m < len(feats['pos_seq_itemID']):
        while True:
            cur_id = random.randint(1,n_item-1)
            if cur_id in pos_seq_items:
                continue
            else:
                break
        neg_seq_id.append(cur_id)
        neg_seq_price.append(item_map[cur_id][2])
        neg_seq_cates.append(item_map[cur_id][1])
        neg_seq_rating.append(rates[random.randint(0,4)])
        m+=1

    #neg_sequence_id = tf.random.shuffle(tf.range(0,n_item,dtype=tf.int64))[:seq_length]
    neg_seq_id = tf.convert_to_tensor(neg_seq_id,dtype=tf.int64)
    neg_seq_price = tf.convert_to_tensor(neg_seq_price,dtype=tf.float32)

    neg_seq_cates = tf.ragged.constant(neg_seq_cates,dtype=tf.int64)
    neg_seq_cates = neg_seq_cates.to_tensor()

    neg_seq_rating = tf.convert_to_tensor(neg_seq_rating,dtype=tf.float32)

    #negtive sampling for target item
    target = feats['pos_itemID'].numpy()
    while True:
        neg_target = random.randint(0,n_item-1)
        if neg_target == target:
            continue
        else:
            break
    neg_cates = item_map[neg_target][1]
    neg_price = item_map[neg_target][2]

    neg_target = tf.convert_to_tensor(neg_target,dtype=tf.int64)
    neg_cates = tf.convert_to_tensor(neg_cates,dtype=tf.int64)
    neg_price = tf.convert_to_tensor(neg_price,dtype=tf.float32)
    
    return (feats['reviewerID'],
    feats['pos_seq_itemID'],feats['pos_seq_cates'],feats['pos_seq_price'],feats['pos_seq_rating'],
    neg_seq_id,neg_seq_cates,neg_seq_price,neg_seq_rating,
    feats['pos_itemID'],feats['pos_cates'],feats['pos_price'],
    neg_target,neg_cates,neg_price)
"""

"""
train_files = [os.path.join(tfr_dir,f"train_{i}.tfrecords") for i in range(num_cores)]
train_ds = tf.data.TFRecordDataset(train_files,num_parallel_reads=num_cores)
train_ds = train_ds.cache().shuffle(20000,reshuffle_each_iteration=True).map(lambda inp : tf.py_function(func=train_parse,inp=[inp],Tout=[
    tf.int64,
    tf.int64,tf.int64,tf.float32,tf.float32,
    tf.int64,tf.int64,tf.float32,tf.float32,
    tf.int64,tf.int64,tf.float32,tf.int64,tf.int64,tf.float32
]),num_parallel_calls=num_cores).map(train_tune,num_parallel_calls=num_cores)

padded_shapes = {
    "reviewerID" : [],
    "pos_seq" : ([None],[None,None],[None],[None]),
    "neg_seq" : ([None],[None,None],[None],[None]),
    "pos_target" : ([],[None],[]),
    "neg_target" : ([],[None],[])
}

train_ds = train_ds.padded_batch(global_batch_size,padded_shapes=padded_shapes)
"""

"""
def test_parse(x):
    parsed = tf.io.parse_sequence_example(x,context_features=ctx_dict,sequence_features=seq_dict)
    feats = {}
    for dic in parsed:
        feats.update(dic)
    for key, value in feats.items():
        if type(value) == tf.sparse.SparseTensor:
            feats[key] = tf.sparse.to_dense(value)

    pos_items = feats['pos_itemID']
    target_length = tf.shape(pos_items)
    candidates = tf.random.uniform(shape=(target_length),minval=1,maxval=n_item,dtype=tf.int64)

    random_shift = tf.random.uniform(shape=(target_length),minval=-1000,maxval=1000,dtype=tf.int64)
    random_shift = pos_items + random_shift
    #larger than 0, 0 is pad
    not_less = tf.cast(tf.greater(random_shift,0),tf.int64)
    #smaller than n_item
    not_greater = tf.cast(tf.less(random_shift,n_item),tf.int64)
    #meet both requirements
    legal_shift = tf.greater((not_less + not_greater),1)
    #if not meet, replace it with candidates
    random_shift = tf.where(legal_shift,random_shift,candidates)
    #candidates as same as pos_seq_items is invalid
    invalid_candidates = tf.equal(pos_items,candidates)
    #if it is invalid, replace candidates with randomshift
    neg_target = tf.where(invalid_candidates,random_shift,candidates)
    neg_price = tf.gather(price_list,indices=neg_target,axis=0)
    neg_cates = tf.gather(cates_list,indices=neg_target,axis=0)


    output = {}
    output['reviewerID'] = feats['reviewerID']
    output['pos_seq'] = (feats['pos_seq_itemID'],feats['pos_seq_cates'],feats['pos_seq_price'],feats['pos_seq_rating'])
    output['pos_target'] = (feats['pos_itemID'],feats['pos_cates'],feats['pos_price'])
    output['neg_target'] = (neg_target,neg_cates,neg_price)
    return output
"""