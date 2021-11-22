import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras.layers import *
from configs import n_cates,n_item,n_reviewer
from tensorflow.keras import losses
from tensorflow.keras import metrics

class AUGRUCell(tf.keras.layers.Layer):
    def __init__(self,
                units,
                activation='tanh',
                recurrent_activation='hard_sigmoid',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                dropout=0.0,
                recurrent_dropout=0.0):
        super(AUGRUCell,self).__init__()
        #last dim of output

        self.units = units

        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.dp = True if dropout!=0.0 else False

        if self.dp:
            self.dropout = {'input_z':Dropout(dropout),
                        'input_r':Dropout(dropout),
                        'input_h':Dropout(dropout)}
        
        self.rdp = True if recurrent_dropout != 0.0 else False
        if self.rdp:
            self.recurrent_dropout = {'states_z':Dropout(recurrent_dropout),
                                  'states_r':Dropout(recurrent_dropout),
                                  'states_h':Dropout(recurrent_dropout)}

        self.built = False
        

    def build(self, input_shape):
        #input shape should be [B,E]
        input_dim = input_shape[-1]
        #[B,E] matmul with kernel [E, Units] ->[B,Units]
        self.kernel = self.add_weight(
            shape = (input_dim, self.units * 3),
            name = 'kernel',
            initializer = self.kernel_initializer,
            trainable = True
        )
        #[B,U] matmul with recurrent kernel [Units,Units] -> [B,Units]
        self.recurrent_kernel = self.add_weight(
            shape = (self.units, self.units * 3),
            name = 'recurrent_kernel',
            initializer = self.recurrent_initializer,
            trainable = True
        )
        #For each operation, single bias is [Units,] vector
        if self.use_bias:
            bias_shape = (2, 3 * self.units)

            self.bias = self.add_weight(
                shape = bias_shape,
                name = 'bias',
                initializer = self.bias_initializer,
                trainable = True
            )
        
        if self.dp:
            for key in self.dropout.keys():
                self.dropout[key].build(input_shape=input_shape)#(None,None,self.units))

        if self.rdp:
            for key in self.recurrent_dropout.keys():
                self.recurrent_dropout[key].build(input_shape=input_shape)#(None,None,self.units))
        self.built = True

    def call(self, inputs, states, cur_alpha, training = None):
        if training == None:
            raise ValueError("Training can not be None, there is no default value for training")
        #inputs [B,E] states [B,U] cur_alpha[B,1]
        if self.use_bias:
            #each is (3*units)
            input_bias, recurrent_bias = tf.unstack(self.bias)

        if self.dp:
            input_z = self.dropout['input_z'](inputs,training=training)
            input_r = self.dropout['input_r'](inputs,training=training)
            input_h = self.dropout['input_h'](inputs,training=training)
        else:
            input_z = inputs
            input_r = inputs
            input_h = inputs

        #[B,E] matmul [E,U] -> [B,U]
        x_z = tf.matmul(input_z,self.kernel[:,:self.units])
        x_r = tf.matmul(input_r,self.kernel[:,self.units:self.units*2])
        x_h = tf.matmul(input_h,self.kernel[:,self.units*2:])

        if self.use_bias:
            x_z = tf.nn.bias_add(x_z, input_bias[:self.units])
            x_r = tf.nn.bias_add(x_r, input_bias[self.units:self.units*2])
            x_h = tf.nn.bias_add(x_h, input_bias[self.units*2:])

        if self.rdp:
            states_z = self.recurrent_dropout['states_z'](states,training=training)
            states_r = self.recurrent_dropout['states_r'](states,training=training)
            states_h = self.recurrent_dropout['states_h'](states,training=training)

        else:
            states_z = states
            states_r = states
            states_h = states

        #[B,U] matmul [U,U] -> [B,U]
        r_z = tf.matmul(states_z,self.recurrent_kernel[:,:self.units])
        r_r = tf.matmul(states_r,self.recurrent_kernel[:,self.units:self.units*2])
        r_h = tf.matmul(states_h,self.recurrent_kernel[:,self.units*2:])
        

        if self.use_bias:
            r_z = tf.nn.bias_add(r_z,recurrent_bias[:self.units])
            r_r = tf.nn.bias_add(r_r,recurrent_bias[self.units:self.units*2])
            r_h = tf.nn.bias_add(r_h,recurrent_bias[self.units*2:])

        #[B,U] + [B,U] -> [B,U]
        #update gate
        #[B,U] * [B,1] broadcast [B,U]
        z = self.recurrent_activation(x_z + r_z)
        z = cur_alpha * z

        r = self.recurrent_activation(x_r + r_r)
        r_h = r * r_h
        hh = self.activation(x_h + r_h)
        # update * cur_state + (1-update) * pre_state
        h = states * (1.0-z) + z * hh

        return h

#augrucell = AUGRUCell(units = 8,dropout=0.1,recurrent_dropout=0.1)
#a = tf.random.uniform(shape=(3,4,5))
#b = tf.random.uniform(shape=(3,4,8))
#alpha = tf.constant(0.5)
#print(augrucell(a,b,alpha,training=True))

class AttentionUpdateGRU(tf.keras.layers.Layer):
    def __init__(self,
                units,
                activation='tanh',
                recurrent_activation='hard_sigmoid',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                dropout=0.,
                recurrent_dropout=0.,
                return_sequences=False,
                **kwargs):
        super(AttentionUpdateGRU,self).__init__(**kwargs)
        self.cell = AUGRUCell(
            units=units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )
        self.return_sequences = return_sequences


    def get_initial_zero_filled_states(self,batch_size,dtype):
        init_state_shape = [batch_size,self.cell.units]
        return tf.zeros(init_state_shape,dtype=dtype)

    def call(self,inputs, alphas, mask, training = None):
        # inputs shape should be [Batch,Time,Emb]
        if mask is None:
            raise ValueError("Only support mask is not None input!")

        if training is None:
            raise ValueError("Must set training Mode to avoid error!")

        #only support input with dim [Batch,Time,EMbedding_dim]
        tf.assert_rank(inputs,3)

        batch_size = tf.shape(inputs)[0]
        cur_dtype = inputs.dtype    #maybe use keras.mixed_precision
        #No, I changed my mind, other places all use tf.float32
        time_length = tf.shape(inputs)[1]

        #time major [B,T,E] - > [T,B,E] 
        inputs = tf.transpose(inputs, perm = [1,0,2])
        #[T,B,1]
        mask = tf.transpose(tf.expand_dims(mask,-1), perm=[1,0,2])
        #[T,B,1]
        alphas = tf.transpose(tf.expand_dims(alphas,-1),perm=[1,0,2])

        #initial_states all zeros
        states = self.get_initial_zero_filled_states(batch_size,cur_dtype)

        def _step_func(cell_inputs, cell_states, cur_alpha, training):
            return self.cell(inputs = cell_inputs, states = cell_states, 
                cur_alpha = cur_alpha, training = training)

        input_ta = tf.TensorArray(dtype=inputs.dtype,
            size=time_length,
            name='inputs_tensor_array',
            clear_after_read=False)

        #Only reserve all hidden_states if return sequences
        if self.return_sequences:
            states_ta = tf.TensorArray(dtype=states.dtype,
                size=time_length,
                name='states_tensor_array',
                clear_after_read=False)
        #just ignore it, set it to zero, because while_loop does not accept None arg
        else:
            states_ta = tf.constant(0)

        mask_ta = tf.TensorArray(dtype=mask.dtype,
            size=time_length,
            name='mask_tensor_array',
            clear_after_read=False)

        alpha_ta = tf.TensorArray(dtype=alphas.dtype,
            size=time_length,
            name='alphas_tensor_array',
            clear_after_read=False)

        #then it is a tensorArray, length is TimeLength, each element with shape [Batch,Emb] or [Batch,1]
        input_ta = input_ta.unstack(inputs)
        mask_ta = mask_ta.unstack(mask)
        alpha_ta = alpha_ta.unstack(alphas)

        time = tf.constant(0,dtype=tf.int32,name='time')

        def _step(time,states,states_ta,training):
            input_t = input_ta.read(time)
            mask_t = mask_ta.read(time)
            alpha_t = alpha_ta.read(time)

            new_states = _step_func(input_t,states,alpha_t,training)
            #[B,1] -> [B,Units]
            mask_t = tf.tile(mask_t,multiples=[1,self.cell.units])
            #if not a pad token, states become new states, otherwise, states does not change
            states = tf.where(mask_t,new_states,states)

            if self.return_sequences:
                states_ta = states_ta.write(time,states)

            return (time + 1, states, states_ta, training)

        _, last_states, states_ta, _ = tf.while_loop(
            cond = lambda time, *_ : time < time_length,
            body = _step,
            loop_vars = (time,states,states_ta,training)
        )

        if self.return_sequences:
            states_ta = states_ta.stack()
            #back to [B,T,E]
            states_ta = tf.transpose(states_ta,perm=[1,0,2])
            return last_states, states_ta
        else:
            return last_states

        

#input_dim = 12
#model = AttentionUpdateGRU(units=8,input_dim=input_dim,dropout=0.1,recurrent_dropout=0.1,return_sequences=True)
#a = tf.random.uniform(shape=(3,4,input_dim))
#b = tf.random.uniform(shape=(4,))
#c = tf.ones(shape=(3,3),dtype=tf.bool)
#c_ = tf.zeros(shape=(3,1),dtype=tf.bool)
#c = tf.concat([c,c_],axis=-1)
#print(model.unroll_call(inputs=a,alphas=b,mask=c,training=False))
#print(model(inputs=a,alphas=b,mask=c,training=False))


class DiceActivation(Layer):
    def __init__(self, axis = -1):
        super(DiceActivation,self).__init__()
        self.axis = axis
        self.bn = BatchNormalization(axis=axis)

    def build(self, input_shape):
        #kernel should be a 1-D Variable with shape (None,)
        kernel_dim = input_shape[self.axis]
        
        self.alphas = self.add_weight(
            shape = (kernel_dim,),
            name = 'alphas',
            initializer = 'glorot_uniform',
            trainable = True
        )

    def call(self, inputs, training=None):
        if training == None:
            raise ValueError("Training must be set either True or False, can not be None")

        inputs = self.bn(inputs,training=training)

        p = tf.sigmoid(inputs)
        
        return self.alphas * (1.0-p) * inputs + p * inputs

#m = DiceActivation()
#tt = tf.random.uniform((3,4,5),dtype=tf.float32)
#print(m(tt))

class AuxiliaryNet(Layer):
    def __init__(self,units_1 = 100, units_2 = 50):
        super(AuxiliaryNet,self).__init__()
        self.bn = BatchNormalization(epsilon=1e-6)
        self.dnn1 = Dense(units=units_1,activation='sigmoid')
        self.dnn2 = Dense(units=units_2,activation='sigmoid')
        self.dnn3 = Dense(units=2,activation='softmax')

    def call(self,inputs):
        bn1 = self.bn(inputs,training=True) #Only used during training mode
        dnn1 = self.dnn1(bn1)
        dnn2 = self.dnn2(dnn1)
        output = self.dnn3(dnn2) + 1e-6
        return output

#m = AuxiliaryNet()
#i = tf.random.uniform((3,5))
#print(m(i))

class DIN_FCN_ATTENTION(Layer):
    def __init__(self, units_1 = 80, units_2 = 40, proj_dim = 32, dropout = 0.1):
        super(DIN_FCN_ATTENTION,self).__init__()
        self.proj1 = Dense(proj_dim)
        self.prelu1 = PReLU(shared_axes=[0,1])
        self.proj2 = Dense(proj_dim)
        self.prelu2 = PReLU(shared_axes=[0,1])
        self.dnn1 = Dense(units = units_1,activation='sigmoid')
        self.dp1 = Dropout(dropout)
        self.dnn2 = Dense(units = units_2,activation='sigmoid')
        self.dp2 = Dropout(dropout)
        self.dnn3 = Dense(units = 1)

    def call(self, target, sequence, mask, training=None):
        if training == None:
            raise ValueError("Training can not be None.")
        # [B,E]
        target = self.proj1(target)
        target = self.prelu1(target)
        # [B,L,E]
        sequence = self.proj2(sequence)
        sequence = self.prelu2(sequence)
        seq_length = tf.shape(sequence)[1]
        # [B,L*E]
        target = tf.tile(target,multiples = [1,seq_length])
        # [B,L,E]
        target = tf.reshape(target,tf.shape(sequence))
        # [B,L,E_]
        concat = tf.concat([target,sequence,target-sequence,target*sequence], axis = -1)
        dnn1 = self.dnn1(concat)
        dnn1 = self.dp1(dnn1,training=training)
        dnn2 = self.dnn2(dnn1)
        dnn2 = self.dp2(dnn2,training=training)
        # [B,L,1]
        dnn3 = self.dnn3(dnn2)
        # [B,L]
        dnn3 = tf.squeeze(dnn3,axis=-1)

        # [B,L]
        all_true = tf.ones_like(mask,dtype=tf.bool)
        mask = tf.not_equal(mask,all_true)
        mask = tf.cast(mask, tf.float32) * -1e9
        dnn3 += mask
        # [B,L]
        attn_alphas = tf.nn.softmax(dnn3)
        return attn_alphas

#m = DIN_FCN_ATTENTION()
#tt = tf.random.uniform((3,8))
#ss = tf.random.uniform((3,4,7))
#mask1 = tf.ones((3,3),tf.bool)
#mask2 = tf.zeros((3,1),tf.bool)
#mask = tf.concat([mask1,mask2],axis=-1)
#print(m(tt,ss,mask))

class FinalFCN(Layer):
    def __init__(self,units_1,units_2,use_dice = True):
        super(FinalFCN,self).__init__()
        self.bn = BatchNormalization(epsilon=1e-6)
        self.dnn1 = Dense(units=units_1)
        if use_dice:
            self.activation1 = DiceActivation()
        else:
            self.activation1 = PReLU(shared_axes=[0])
        self.dnn2 = Dense(units=units_2)
        if use_dice:
            self.activation2 = DiceActivation()
        else:
            self.activation2 = PReLU(shared_axes=[0])
        self.dnn3 = Dense(1)
    def call(self,inputs,training=None):
        if training == None:
            raise ValueError("Training should not be None!")
        bn = self.bn(inputs,training=training)
        dnn1 = self.dnn1(bn)
        dnn1 = self.activation1(dnn1)
        dnn2 = self.dnn2(dnn1)
        dnn2 = self.activation2(dnn2)
        dnn3 = self.dnn3(dnn2)
        return dnn3

class DIEN(tf.keras.Model):
    def __init__(self, 
                embedding_dim = 32, useNorm=True, pad_idx = 0,
                n_user = n_reviewer,n_item = n_item,n_cates = n_cates,
                gru_dropout=0.1, gru_units = 64,
                aux_units1=128,aux_units2=64,
                attn_units1 = 128, attn_units2 = 64, attn_proj_dim = 32, attn_dropout = 0.1,
                augru_units=128, #set it relatively larger, to make seq information more important when concat
                augru_dropout = 0.1, augru_rec_dropout = 0.1,
                final_units1 = 128, final_units2 = 64,
                aux_loss_weight = 0.5, norm_loss_weight = 1e-4
                ):
        super(DIEN,self).__init__()

        self.user_emb = Embedding(input_dim=n_user,output_dim=embedding_dim,
            embeddings_initializer='glorot_uniform',mask_zero=True)
        self.item_emb = Embedding(input_dim=n_item,output_dim=embedding_dim,
            embeddings_initializer='glorot_uniform',mask_zero=True)
        self.cate_emb = Embedding(input_dim=n_cates,output_dim=embedding_dim,
            embeddings_initializer='glorot_uniform',mask_zero=True)

        self.normal_gru = GRU(gru_units,dropout=gru_dropout,return_sequences=True)

        self.aux_net = AuxiliaryNet(aux_units1,aux_units2)
        
        self.din_fcn_attn = DIN_FCN_ATTENTION(attn_units1,attn_units2,proj_dim=attn_proj_dim,dropout=attn_dropout)

        self.augru = AttentionUpdateGRU(units=augru_units,dropout=augru_dropout,recurrent_dropout=augru_rec_dropout)

        self.final_fcn = FinalFCN(units_1=final_units1,units_2=final_units2)

        self.useNorm = useNorm
        
        self.pad = pad_idx

        self.aux_loss_weight = aux_loss_weight

        self.norm_loss_weight = norm_loss_weight

        self.binary_cross_entropy_loss = losses.BinaryCrossentropy(from_logits=True,reduction='none')

        #self.training_auc = metrics.AUC(from_logits=True)
        
    def call(self,inputs,training=None):
        if training == None:
            raise ValueError("Training must be set")

        userID = inputs['reviewerID']
        pos_seq = inputs['pos_seq']
        if training:
            neg_seq = inputs['neg_seq']
        pos_target = inputs['pos_target']
        neg_target = inputs['neg_target']

        userID = self.user_emb(userID)
        pos_seq_ids = pos_seq[0]
        seq_mask = tf.not_equal(pos_seq_ids, self.pad)


        if self.useNorm and training:
            normloss = self.normLoss(userID)

            pos_seq, tmploss = self.concat_seq_emb(pos_seq,training=True)
            normloss += tmploss
            neg_seq, tmploss = self.concat_seq_emb(neg_seq,training=True)
            normloss += tmploss
            pos_target, tmploss = self.target_emb(pos_target,training=True)
            normloss += tmploss
            neg_target, tmploss = self.target_emb(neg_target,training=True)

        else:
            pos_seq = self.concat_seq_emb(pos_seq,training=False)
            if training:
                neg_seq = self.concat_seq_emb(neg_seq,training=False)
            pos_target = self.target_emb(pos_target,training=False)
            neg_target = self.target_emb(neg_target,training=False)


        if training:
            pos_augru_output, neg_augru_output, auxloss = self.rnn_attention_modules((pos_seq, neg_seq, 
                pos_target, neg_target),mask=seq_mask,training=training)
        else:
            pos_augru_output, neg_augru_output = self.rnn_attention_modules((pos_seq, 
                pos_target, neg_target),mask=seq_mask,training=training)
        
        #[B,D]
        pos_all = tf.concat([pos_augru_output,userID,pos_target],axis=-1)
        neg_all = tf.concat([neg_augru_output,userID,neg_target],axis=-1)
        #[B,1]
        pos_logits = self.final_fcn(inputs=pos_all,training=training)
        neg_logits = self.final_fcn(inputs=neg_all,training=training)
        #[2B]
        logits = tf.squeeze(tf.concat([pos_logits,neg_logits],axis=0),axis=-1)
        
        #tf.assert_rank(logits,1)

        ones = tf.ones(shape=tf.shape(pos_logits))
        zeros = tf.zeros(shape=tf.shape(neg_logits))
        labels = tf.squeeze(tf.concat([ones,zeros],axis=0))

        #tf.assert_rank(labels,1)
        if training:
            ctr_loss = self.binary_cross_entropy_loss(y_true=labels,y_pred=logits)
            
            loss = tf.reduce_mean(ctr_loss) + self.aux_loss_weight * auxloss

            if self.useNorm:
                loss += self.norm_loss_weight * normloss 

        #binary_accuracy = tf.equal(tf.round(tf.sigmoid(logits)),labels)

        #binary_accuracy = tf.equal(tf.round(tf.sigmoid(logits)),labels)
        #binary_accuracy = tf.reduce_mean(tf.cast(binary_accuracy,dtype=tf.float32))
        #binary_accuracy = tf.sigmoid(logits)

        #self.training_auc.update_state(y_true=labels,y_pred=logits)
        #training_auc = self.training_auc.result()
        #self.training_auc.reset_state()
        if training:
            #return loss,  binary_accuracy , training_auc
            return loss, logits, labels
        else:
            #return binary_accuracy, training_auc
            return None


        
    def rnn_attention_modules(self,inputs,mask,training=None):
        if training == None:
            raise ValueError("Training should not be None!")
        if training:
            pos_seq, neg_seq, pos_target, neg_target = inputs
        else:
            pos_seq, pos_target, neg_target = inputs
        #hidden states sequence of normal gru
        hidden_states_seq = self.normal_gru(pos_seq, mask=mask,training = training)
        #auxiliary loss 
        if training:
            auxloss = self.auxnetLoss(hidden_states=hidden_states_seq,
                pos_seq=pos_seq,neg_seq=neg_seq,mask=mask)
        #pos_target / neg_target with pos_hidden_states to get alphas
        pos_alphas = self.din_fcn_attn(pos_target,hidden_states_seq,mask=mask, training=training)
        neg_alphas = self.din_fcn_attn(neg_target,hidden_states_seq,mask=mask, training=training)

        pos_augru_output = self.augru(inputs = hidden_states_seq, alphas = pos_alphas , mask = mask, training = training)
        neg_augru_output = self.augru(inputs = hidden_states_seq, alphas = neg_alphas , mask = mask, training = training)

        if training:
            return pos_augru_output, neg_augru_output, auxloss
        else:
            return pos_augru_output, neg_augru_output



    def auxnetLoss(self,hidden_states,pos_seq,neg_seq,mask):
        #Next emb supervise current hid
        hid = hidden_states[:,:-1,:]
        pos = pos_seq[:,1:,:]
        neg = neg_seq[:,1:,:]
        #mask is for emb, if emb is pad, loss for current timestamp is not counted
        msk = mask[:,1:]
        msk = tf.cast(msk,dtype=hid.dtype)

        pos_input = tf.concat([hid,pos],axis=-1)
        neg_input = tf.concat([hid,neg],axis=-1)

        #let 0 stands for true class
        pos = self.aux_net(pos_input)[:,:,0]
        neg = self.aux_net(neg_input)[:,:,1]

        #tf.assert_rank(pos,2)

        pos_istrue_prob = tf.math.log(pos) * msk * -1.0
        neg_isfalse_prob = tf.math.log(neg) * msk * -1.0

        auxloss = tf.reduce_mean(pos_istrue_prob + neg_isfalse_prob)
        
        #tf.assert_greater(auxloss,0.0)

        return auxloss


    def concat_seq_emb(self,seq,training=None):
        if training == None:
            raise ValueError("Training can not be None.")
        #          price[B,L]  rating[B,L]
        itemID, cates, price, rating = seq
        price = tf.expand_dims(price,axis=-1)
        rating = tf.expand_dims(rating,axis=-1)
        # [B,L,E]
        itemID = self.item_emb(itemID)
        cates_mask = tf.expand_dims(tf.cast(tf.not_equal(cates,self.pad),itemID.dtype),axis=-1)
        cates = self.cate_emb(cates)
        cates = tf.multiply(cates,cates_mask)
        #tf.assert_rank(cates,4)
        # [B,L,E]
        cates = tf.reduce_sum(cates,axis=2)

        output = tf.concat([itemID,cates,price,rating],axis=-1)

        if training:
            l2loss = self.normLoss(output)
            return output,l2loss
        else:
            return output

    def target_emb(self,seq,training=None):
        if training == None:
            raise ValueError("Training can not be None.")
        itemID, cates, price = seq
        price = tf.expand_dims(price,axis=-1)
        itemID = self.item_emb(itemID)
        cates_mask = tf.expand_dims(tf.cast(tf.not_equal(cates,self.pad),itemID.dtype),axis=-1)
        cates = self.cate_emb(cates)
        cates = tf.multiply(cates,cates_mask)
        #tf.assert_rank(cates,3)
        cates = tf.reduce_sum(cates,axis=1)

        output = tf.concat([itemID,cates,price],axis=-1)
        if training:
            l2loss = self.normLoss(output)
            return output,l2loss
        else:
            return output
        

    def normLoss(self,embs,hurdle=1.0):
        l2norm = tf.pow(embs,2)
        l2norm = tf.reduce_sum(l2norm,axis=-1,keepdims=True)
        zeros = tf.zeros_like(l2norm,dtype=l2norm.dtype)
        return tf.reduce_mean(tf.reduce_max(
            tf.concat([l2norm-hurdle,zeros],axis=-1),axis=-1
        ))








"""
import time
for step,inp in enumerate(train_ds):
    if step == 0:
        start = time.time()
    loss = train_step(inp)
    if step == 19:
        end = time.time()
        break
print(f"{(end-start)/20} seconds per batch (1024)")
print("========================")
"""
"""
Naive Python Loop negsampling cost 4.98s per batch(1024)
"""
"""
Random Negsampling cost 0.481s per batch(10240)
"""

"""
class DiceActivation(Layer):
    def __init__(self, axis = -1):
        super(DiceActivation,self).__init__()
        self.axis = axis

    def build(self, input_shape):
        #kernel should be a 1-D Variable with shape (None,)
        kernel_dim = input_shape[self.axis]
        
        self.alphas = self.add_weight(
            shape = (kernel_dim,),
            name = 'alphas',
            initializer = 'glorot_uniform',
            trainable = True
        )

        self.input_rank = len(input_shape)
        #preserve self.axis, other dimensions will be reducted
        self.reduction_axes = list(range(self.input_rank))
        del self.reduction_axes[self.axis]

        #expand_dims to make the rank as same as inputs
        self.broadcast_shape = [1] * self.input_rank
        self.broadcast_shape[self.axis] = kernel_dim

    def call(self, inputs, epsilon = 1e-9):
        # Get mean value for each element on self.axis across all other axes
        mean = tf.reduce_mean(inputs, axis=self.reduction_axes)
        # shape like [1,1,...,kernel_dim,1,...1,], all 1 besides self.axis
        broadcast_mean = tf.reshape(mean, self.broadcast_shape)
        # stddev
        std = tf.reduce_mean(tf.square(inputs - broadcast_mean) + epsilon, axis=self.reduction_axes)
        std = tf.sqrt(std)
        # shape like [1,1,...,kernel_dim,1,...1,], all 1 besides self.axis
        broadcast_std = tf.reshape(std,self.broadcast_shape)
        # x_hat = (x-mean) / sqrt(std)
        normed_inputs = (inputs - broadcast_mean) / broadcast_std
        # p(x) = sigmoid(x_hat)
        p = tf.sigmoid(normed_inputs)
        
        return self.alphas * (1.0-p) * inputs + p * inputs
"""


"""
        if training:
            userID = inputs['reviewerID']
            pos_seq = inputs['pos_seq']
            neg_seq = inputs['neg_seq']
            pos_target = inputs['pos_target']
            neg_target = inputs['neg_target']

            userID = self.user_emb(userID)
            pos_seq_ids = pos_seq[0]
            seq_mask = tf.not_equal(pos_seq_ids, self.pad)


            if self.useNorm:
                normloss = self.normLoss(userID)

                pos_seq, tmploss = self.concat_seq_emb(pos_seq,training=True)
                normloss += tmploss
                neg_seq, tmploss = self.concat_seq_emb(neg_seq,training=True)
                normloss += tmploss
                pos_target, tmploss = self.target_emb(pos_target,training=True)
                normloss += tmploss
                neg_target, tmploss = self.target_emb(neg_target,training=True)

            else:
                pos_seq = self.concat_seq_emb(pos_seq,training=False)
                neg_seq = self.concat_seq_emb(neg_seq,training=False)
                pos_target = self.target_emb(pos_target,training=False)
                neg_target = self.target_emb(neg_target,training=False)


            pos_augru_output, neg_augru_output, auxloss = self.rnn_attention_modules((pos_seq, neg_seq, 
                pos_target, neg_target),mask=seq_mask,training=True)
            
            #[B,D]
            pos_all = tf.concat([pos_augru_output,userID,pos_target],axis=-1)
            neg_all = tf.concat([neg_augru_output,userID,neg_target],axis=-1)
            #[B,1]
            pos_logits = self.final_fcn(inputs=pos_all,training=training)
            neg_logits = self.final_fcn(inputs=neg_all,training=training)
            #[2B]
            logits = tf.squeeze(tf.concat([pos_logits,neg_logits],axis=0),axis=-1)
            
            #tf.assert_rank(logits,1)

            ones = tf.ones(shape=tf.shape(pos_logits))
            zeros = tf.zeros(shape=tf.shape(neg_logits))
            labels = tf.squeeze(tf.concat([ones,zeros],axis=0))

            #tf.assert_rank(labels,1)
            
            ctr_loss = self.binary_cross_entropy_loss(y_true=labels,y_pred=logits)
            
            loss = tf.reduce_mean(ctr_loss) + self.aux_loss_weight * auxloss

            if self.useNorm:
                loss += self.norm_loss_weight * normloss 

            binary_accuracy = tf.equal(tf.round(tf.sigmoid(logits)),labels)

            #binary_accuracy = tf.equal(tf.round(tf.sigmoid(logits)),labels)
            binary_accuracy = tf.reduce_mean(tf.cast(binary_accuracy,dtype=tf.float32))
            #binary_accuracy = tf.sigmoid(logits)

            self.training_auc.update_state(y_true=labels,y_pred=logits)
            training_auc = self.training_auc.result()
            self.training_auc.reset_state()

            return loss, binary_accuracy , training_auc

        else:
            userID = inputs['reviwerID']
            pos_seq = inputs['pos_seq']
            pos_target = inputs['pos_target']
            neg_target = inputs['neg_target']
            
            userID = self.user_emb(userID)

            pos_seq_ids = pos_seq[0]
            seq_mask = tf.not_equal(pos_seq_ids, self.pad)

            pos_seq = self.concat_seq_emb(pos_seq,training=False)
            pos_target = self.target_emb(pos_target,training=False)
            neg_target = self.target_emb(neg_target,training=False)

            pos_augru_output, neg_augru_output = self.rnn_attention_modules((pos_seq,pos_target,neg_target),
                mask=seq_mask,training=False)

            pos_all = tf.concat([augru_output,userID,pos_target],axis=-1)

            logits = self.final_fcn(pos_all,training=training)

            logits = tf.squeeze(logits,axis=-1)

            labels = tf.ones(shape=tf.shape(logits))

            binary_accuracy = tf.equal(tf.round(tf.sigmoid(logits)),labels)
            binary_accuracy = tf.reduce_mean(tf.cast(binary_accuracy,dtype=tf.float32))

            return logits, binary_accuracy
"""