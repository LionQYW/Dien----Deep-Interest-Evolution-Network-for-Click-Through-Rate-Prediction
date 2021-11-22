import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras.layers import *




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
        
        self.rdp = True if recurrent_dropout!=0.0 else False
        if self.rdp:
            self.recurrent_dropout = {'states_z':Dropout(recurrent_dropout),
                                  'states_r':Dropout(recurrent_dropout),
                                  'states_h':Dropout(recurrent_dropout)}

        self.state_size = self.units
        self.output_size = self.units
        self.built = False
        

    def build(self, input_shape):

        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape = (input_dim, self.units * 3),
            name = 'kernel',
            initializer = self.kernel_initializer,
            trainable = True
        )
        
        self.recurrent_kernel = self.add_weight(
            shape = (self.units, self.units * 3),
            name = 'recurrent_kernel',
            initializer = self.recurrent_initializer,
            trainable = True
        )

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
                self.dropout[key].build(input_shape=(None,None,self.units))
        if self.rdp:
            for key in self.recurrent_dropout.keys():
                self.recurrent_dropout[key].build(input_shape=(None,None,self.units))
        self.built = True

    def call(self, inputs, states, cur_alpha, training = None):

        if self.use_bias:
            input_bias, recurrent_bias = tf.unstack(self.bias)

        if self.dp:
            input_z = self.dropout['input_z'](inputs,training=training)
            input_r = self.dropout['input_r'](inputs,training=training)
            input_h = self.dropout['input_h'](inputs,training=training)
        else:
            input_z = inputs
            input_r = inputs
            input_h = inputs

        x_z = tf.matmul(input_z,self.kernel[:,:self.units])
        x_r = tf.matmul(input_r,self.kernel[:,self.units:self.units*2])
        x_h = tf.matmul(input_h,self.kernel[:,self.units*2:])

        if self.use_bias:
            x_z = tf.nn.bias_add(x_z, input_bias[:self.units])
            x_r = tf.nn.bias_add(x_r, input_bias[self.units:self.units*2])
            x_h = tf.nn.bias_add(x_h, input_bias[self.units*2:])

        if self.rdp:
            states_z = self.recurrent_dropout['states_z'](states)
            states_r = self.recurrent_dropout['states_r'](states)
            states_h = self.recurrent_dropout['states_h'](states)

        else:
            states_z = states
            states_r = states
            states_h = states

        r_z = tf.matmul(states_z,self.recurrent_kernel[:,:self.units])
        r_r = tf.matmul(states_r,self.recurrent_kernel[:,self.units:self.units*2])
        r_h = tf.matmul(states_h,self.recurrent_kernel[:,self.units*2:])
        
        if self.use_bias:
            r_z = tf.nn.bias_add(r_z,recurrent_bias[:self.units])
            r_r = tf.nn.bias_add(r_r,recurrent_bias[self.units:self.units*2])
            r_h = tf.nn.bias_add(r_h,recurrent_bias[self.units*2:])

        z = self.recurrent_activation(x_z + r_z)
        z = cur_alpha * z

        r = self.recurrent_activation(x_r + r_r)
        r_h = r * r_h
        hh = self.activation(x_h + r_h)

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
                input_dim,
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
        #For check purpose, ignore it
        #Just the hidden_dim from inputs, e.g last dimension size of input
        self.input_dim = input_dim
        self.implementation_and_mode_check()



    def get_initial_zero_filled_states(self,batch_size,dtype):
        init_state_size = [batch_size,self.cell.state_size]
        return tf.zeros(init_state_size,dtype=dtype)

    #ckeck if tf.functino mode and implementation of unroll and non-unroll affect the result
    def implementation_and_mode_check(self):
        b, t, e = 6,8,self.input_dim
        test_inputs = tf.random.uniform(shape=(b,t,e))
        test_alpha = tf.random.uniform(shape=(t,))
        mask_1 = tf.ones(shape=(b,t-2),dtype=tf.bool)
        mask_2 = tf.ones(shape=(b,2),dtype=tf.bool)
        mask = tf.concat((mask_1,mask_2),axis=1)

        tmp = self.return_sequences
        self.return_sequences = True

        last_states_1, sequence_1 = self.call(test_inputs,test_alpha,mask,training=False)
        last_states_2, sequence_2 = self.unroll_call(test_inputs,test_alpha,mask,training=False)

        res_1 = tf.reduce_sum(tf.cast(tf.not_equal(last_states_1,last_states_2),dtype=tf.float32))
        res_2 = tf.reduce_sum(tf.cast(tf.not_equal(sequence_1,sequence_2),dtype=tf.float32))
        res = res_1 + res_2
        if res != 0.0:
            raise ValueError("Error implementation occur!")

        self.return_sequences = tmp

    @tf.function
    def call(self,inputs, alphas, mask = None, training = None):
        if mask is None:
            raise ValueError("Only support mask is not None input!")

        if training is None:
            raise ValueError("Must set training Mode to avoid error!")

        #only support input with dim [Batch,Time,EMbedding_dim]
        tf.assert_rank(inputs,3)

        batch_size = tf.shape(inputs)[0]
        cur_dtype = inputs.dtype    #maybe use keras.mixed_precision
        time_length = tf.shape(inputs)[1]

        #time major [B,T,E] - > [T,B,E] 
        inputs = tf.transpose(inputs, perm = [1,0,2])
        #[T,B,1]
        mask = tf.transpose(tf.expand_dims(mask,-1), perm=[1,0,2])

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

        input_ta = input_ta.unstack(inputs)
        mask_ta = mask_ta.unstack(mask)
        alpha_ta = alpha_ta.unstack(alphas)

        time = tf.constant(0,dtype=tf.int32,name='time')

        def _step(time,states,states_ta,training):
            input_t = input_ta.read(time)
            mask_t = mask_ta.read(time)
            alpha_t = alpha_ta.read(time)

            new_states = _step_func(input_t,states,alpha_t,training)

            mask_t = tf.tile(mask_t,multiples=[1,self.cell.units])
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


    #implement GRU based on python list, not applicable when use tf.function
    #For test purpose, can be ignored
    def unroll_call(self,inputs, alphas, mask = None, training = None):
        if mask is None:
            raise ValueError("Only support mask is not None input!")

        #only support input with dim [Batch,Time,EMbedding_dim]
        tf.assert_rank(inputs,3)

        batch_size = tf.shape(inputs)[0]
        cur_dtype = inputs.dtype    #maybe use keras.mixed_precision
        time_length = tf.shape(inputs)[1]

        #time major [B,T,E] - > [T,B,E] 
        inputs = tf.transpose(inputs, perm = [1,0,2])
        #[T,B,1]
        mask = tf.transpose(tf.expand_dims(mask,-1), perm=[1,0,2])

        #initial_states all zeros
        states = self.get_initial_zero_filled_states(batch_size,cur_dtype)

        def _step_func(cell_inputs, cell_states, cur_alpha, training):
            return self.cell(inputs = cell_inputs, states = cell_states, 
                cur_alpha = cur_alpha, training = training)

        successive_states = []
        #list of tensor of size [B,E]
        input_list = tf.unstack(inputs)
        # [B,1]
        mask_list = tf.unstack(mask)
        #list of scalar tensor
        alpha_list = tf.unstack(alphas)

        for i in range(time_length):
            inp_t = input_list[i]
            mask_t = mask_list[i]
            alpha_t = alpha_list[i]
            #[B,E]
            tiled_mask_t = tf.tile(mask_t, multiples=[1,self.cell.units])
            #[B,E]
            new_states = _step_func(inp_t, states, alpha_t, training)

            if not successive_states:
                pre_states = tf.zeros_like(new_states)
            else:
                pre_states = successive_states[-1]

            states = tf.where(tiled_mask_t, new_states, pre_states)
            successive_states.append(states)
        
        last_states = successive_states[-1]
        
        if self.return_sequences:
            sequence = tf.stack(successive_states)
            sequence = tf.transpose(sequence,perm=[1,0,2])
            return last_states, sequence
        else:
            return last_states
        

input_dim = 12
model = AttentionUpdateGRU(units=8,input_dim=input_dim,dropout=0.1,recurrent_dropout=0.1,return_sequences=True)
a = tf.random.uniform(shape=(3,4,input_dim))
b = tf.random.uniform(shape=(4,))
c = tf.ones(shape=(3,3),dtype=tf.bool)
c_ = tf.zeros(shape=(3,1),dtype=tf.bool)
c = tf.concat([c,c_],axis=-1)
print(model.unroll_call(inputs=a,alphas=b,mask=c,training=False))
print(model(inputs=a,alphas=b,mask=c,training=False))


