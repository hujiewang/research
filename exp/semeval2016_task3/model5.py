
from reader import Reader,TRAIN,TEST,DEV
from ev2 import eval_reranker
from preprocess import preprocess
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phrases
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    AdaptiveETA, FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell
import numpy as np


class Model(object):
    def __init__(self):
        self.session = tf.Session()
        '''
        Training parameters:
        '''

        self.w2v_dim=30
        self.num_feature=400
        self.batch_size=32
        self.num_epoch=10000
        self.num_hidden_1=50
        self.num_hidden_2=3

        self.number_of_layers=1

        #self.max_len = 50
        self.max_len_title=6
        self.max_len_body=38

        #self.w2v_model=Word2Vec.load_word2vec_format('./data/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
        self.w2v_model=Word2Vec.load('data/word2vec/w2v.model')
        self.index2word_set = set(self.w2v_model.index2word)
        #self.bigram = None
        #self.trigram =None
        self.bigram=Phrases.load('./data/bigram.dat')
        self.trigram=Phrases.load('./data/trigram.dat')

        # Model
        self.input_0=tf.placeholder(tf.float32,[self.max_len_title,self.batch_size,self.w2v_dim])
        self.input_1=tf.placeholder(tf.float32,[self.max_len_title,self.batch_size,self.w2v_dim])
        self.input_0_=tf.placeholder(tf.float32,[self.max_len_body,self.batch_size,self.w2v_dim])
        self.input_1_=tf.placeholder(tf.float32,[self.max_len_body,self.batch_size,self.w2v_dim])

        self.dropout_input = tf.placeholder(tf.float32)
        self.dropout_hidden = tf.placeholder(tf.float32)

        self.target = tf.placeholder(tf.float32, [self.batch_size, 3])

        input_0=array_ops.unpack(self.input_0)
        input_1=array_ops.unpack(self.input_1)
        input_0_=array_ops.unpack(self.input_0_)
        input_1_=array_ops.unpack(self.input_1_)


        def _rnn(inputs, reverse=False):
            with tf.variable_scope("GRU_RNN") as scope:
                cell=rnn_cell.GRUCell(self.w2v_dim)
                cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_input)
                stacked_cell = rnn_cell.MultiRNNCell([cell] * self.number_of_layers)
                state = stacked_cell.zero_state(self.batch_size, tf.float32)
                if reverse:
                    inputs=reversed(inputs)
                for time, input_ in enumerate(inputs):
                    if time > 0: scope.reuse_variables()
                    output, state = stacked_cell(input_, state)
                return state

        with tf.variable_scope('Feature_Generator') as scope:
            state_0 = _rnn(input_0)
            scope.reuse_variables()
            state_1 = _rnn(input_1)
            state_0_ = _rnn(input_0_)
            state_1_ = _rnn(input_1_)
        '''
        with tf.variable_scope('Feature_Generator_body') as scope:
            state_0_ = _rnn(input_0_)
            scope.reuse_variables()
            state_1_ = _rnn(input_1_)
        '''
        '''
        with tf.variable_scope('Feature_Generator_body_reverse') as scope:
            state_0_reverse = _rnn(input_0_, reverse=True)
            scope.reuse_variables()
            state_1_reverse = _rnn(input_1_, reverse=True)
        '''

        '''
        with tf.variable_scope('Feature_Generator_title') as scope:
            state_0 = _rnn(input_0)
            scope.reuse_variables()
            state_1 = _rnn(input_1)

        with tf.variable_scope('Feature_Generator_body') as scope:
            state_0_ = _rnn(input_0_)
            scope.reuse_variables()
            state_1_ = _rnn(input_1_)


        # state=tf.concat(1,[tf.abs(tf.sub(state_0,state_1)),tf.mul(state_0,state_1),
        #                   tf.abs(tf.sub(state_0_,state_1_)),tf.mul(state_0_,state_1_)])


        # state=tf.concat(1,[state_0,state_1, state_0_, state_1_])
        # state = tf.ones([32,10])

        # state=tf.concat(1,[tf.abs(tf.sub(state_0,state_1)),tf.mul(state_0,state_1)])
        '''

         # 2-layer NN
        with tf.variable_scope("NN", initializer=tf.random_uniform_initializer(-1.0,1.0)):
            self.W_mul = tf.get_variable("W_mul", [state_0_.get_shape()[1]*2,self.num_hidden_1])
            self.W_sub = tf.get_variable("W_sub", [state_0_.get_shape()[1]*2,self.num_hidden_1])
            self.b = tf.get_variable("b", [self.num_hidden_1])

            self.W_softmax=tf.get_variable("W_softmax", [self.num_hidden_1,self.num_hidden_2])
            self.b_softmax = tf.get_variable("b_softmax", [self.num_hidden_2])

        # h_mul = tf.mul(state_0,state_1)
        # h_sub = tf.abs(tf.sub(state_0,state_1))
        h_mul = tf.concat(1,[tf.mul(state_0,state_1),tf.mul(state_0_,state_1_)])
        h_sub = tf.concat(1,[tf.abs(tf.sub(state_0,state_1)),tf.abs(tf.sub(state_0_,state_1_))])

        y_1 = tf.nn.sigmoid(tf.matmul(h_mul, self.W_mul)+tf.matmul(h_sub, self.W_sub)+self.b)
        y_2 = tf.matmul(y_1, self.W_softmax)+self.b_softmax

        # regularizers = (tf.nn.l2_loss(self.W_1) + tf.nn.l2_loss(self.b_1)+tf.nn.l2_loss(self.W_2) + tf.nn.l2_loss(self.b_2))

        '''
        state_0_title_normalized = tf.nn.l2_normalize(state_0, 1)
        state_1_title_normalized = tf.nn.l2_normalize(state_1, 1)
        state_0_body_normalized = tf.nn.l2_normalize(state_0_, 1)
        state_1_body_normalized = tf.nn.l2_normalize(state_1_, 1)

        dist_title_ = tf.mul(state_0_title_normalized, state_1_title_normalized)
        dist_body_ = tf.mul(state_0_body_normalized, state_1_body_normalized)s

        dist_title=tf.reduce_sum(dist_title_, 1, keep_dims=True)
        dist_body=tf.reduce_sum(dist_body_, 1, keep_dims=True)

        feature = tf.concat(1, [dist_title,dist_body])

        with tf.variable_scope("log_reg", initializer=tf.random_uniform_initializer()):
             self.W = tf.get_variable("W", [feature.get_shape()[1],3])
             self.b = tf.get_variable("b", [3])

        y_2 = tf.matmul(feature, self.W)+self.b
        '''
        '''
        with tf.variable_scope("log_reg", initializer=tf.random_uniform_initializer()):
            self.W_1 = tf.get_variable("W_1", [state.get_shape()[1],self.num_hidden_1])
            self.b_1 = tf.get_variable("b_1", [self.num_hidden_1])
            self.W_2 = tf.get_variable("W_2", [self.num_hidden_1,self.num_hidden_2])
            self.b_2 = tf.get_variable("b_2", [self.num_hidden_2])
        '''
        '''
        # Create model
        def multilayer_perceptron(_X, _weights, _biases):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with RELU activation
            return tf.matmul(layer_2, _weights['out']) + _biases['out']

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([10, 10])),
            'h2': tf.Variable(tf.random_normal([10, 5])),
            'out': tf.Variable(tf.random_normal([5, 3]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([10])),
            'b2': tf.Variable(tf.random_normal([5])),
            'out': tf.Variable(tf.random_normal([3]))
        }
        # Construct model
        self.y_pred = multilayer_perceptron(state, weights, biases)

        # Define loss and optimizer
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_pred, self.target)) # Softmax loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.cross_entropy) # Adam Optimizer
        '''

        # self.W = tf.Variable(tf.zeros([10, 3]))
        # self.b = tf.Variable(tf.zeros([3]))
        # y_1 = tf.sigmoid(tf.matmul(state, self.W_1)+self.b_1)
        # y_2 = tf.sigmoid(tf.matmul(y_1, self.W_2)+self.b_2)
        # self.y_pred = tf.nn.softmax(tf.nn.sigmoid(tf.add(tf.matmul(state, self.W),self.b)))
        self.y_pred=tf.nn.softmax(y_2)
        # self.y_pred = tf.nn.softmax(tf.nn.sigmoid(tf.matmul(state, self.W_1)+self.b_1))
        self.cross_entropy = -tf.reduce_mean(self.target*tf.log(self.y_pred))
        # self.optimizer = tf.train.AdamOptimizer().minimize(self.cross_entropy)
        # self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.cross_entropy)
        # self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)
        # self.gradstep = self.optimizer.compute_gradients(self.cross_entropy)


        # Optimizer.

        global_step = tf.Variable(0)
        # optimizer = tf.train.GradientDescentOptimizer(0.1)
        optimizer = tf.train.AdagradOptimizer(0.1)
        gradients, v = zip(*optimizer.compute_gradients(self.cross_entropy))
        gradients, _ = tf.clip_by_global_norm(gradients, 10)
        self.optimizer= optimizer.apply_gradients(zip(gradients, v), global_step=global_step)



        print('Model has been built!')

    def fit(self, X, y):

        _, loss, y_pred=self.session.run([self.optimizer, self.cross_entropy, self.y_pred],{self.input_0: X[0],
                                                                        self.input_1: X[1],
                                                                        self.input_0_: X[2],
                                                                        self.input_1_: X[3],
                                                                        self.dropout_input: 0.2,
                                                                        self.dropout_hidden: 0.5,
                                                                        self.target: y})

        # grads = self.session.run([grad for grad, _ in self.gradstep],{self.input_0: X[0],self.input_1: X[1],self.target: y})

        return loss

    def predict(self, X):
        y_pred =self.session.run([self.y_pred],{self.input_0: X[0],
                                                self.input_1: X[1],
                                                self.input_0_: X[2],
                                                self.input_1_: X[3],
                                                self.dropout_input: 1.0,
                                                self.dropout_hidden: 1.0,})
        return y_pred[0]

    def getWordVectorFeatures(self, text, title=False):
        words = text.split()
        max_len=self.max_len_title if title else self.max_len_body
        featureVec=np.zeros((max_len,1,self.w2v_dim))

        start = max(max_len-len(words),0)
        '''
        for i in range(len(words)):
            if start+i==max_len:break
            word=words[i]
            if word in self.index2word_set:
                featureVec[start+i,0,:]=self.w2v_model[word]
        '''
        k=0
        for i in range(len(words)):
            if i==max_len:break
            word=words[i]
            if word in self.index2word_set:
                featureVec[k,0,:]=self.w2v_model[word]
                k+=1

        return featureVec, len(words)


    def getFeature(self, ori_q,rel_q):
        # ori_q[0]=preprocess(ori_q[0])
        # rel_q[0]=preprocess(rel_q[0])
        ori_q[0]=preprocess(ori_q[0],bigram=self.bigram,trigram=self.trigram, no_stopwords=True)
        rel_q[0]=preprocess(rel_q[0],bigram=self.bigram,trigram=self.trigram, no_stopwords=True)
        ori_q[1]=preprocess(ori_q[1],bigram=self.bigram,trigram=self.trigram, no_stopwords=True)
        rel_q[1]=preprocess(rel_q[1],bigram=self.bigram,trigram=self.trigram, no_stopwords=True)

        word2vec_q_subject, q_len=self.getWordVectorFeatures(ori_q[0], title=True)
        word2vec_rel_q_subject, rel_q_len=self.getWordVectorFeatures(rel_q[0], title=True)

        word2vec_q_body, q_len=self.getWordVectorFeatures(ori_q[1])
        word2vec_rel_q_body, rel_q_len=self.getWordVectorFeatures(rel_q[1])

        return [word2vec_q_subject, word2vec_rel_q_subject, word2vec_q_body, word2vec_rel_q_body]


    def prepareData(self,data):
        size=0
        for i in range(len(data)):
            size+=(len(data[i])/2)-1

        X=[]
        y=[]
        meta=[]

        c=0
        pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(data)).start()
        for i in range(len(data)):
            samples = data[i]

            ori_q_id=samples[0]['ORGQ_ID']
            ori_q=samples[1]

            for j in range(2,len(samples),2):

                rel_q_id=samples[j]['RELQ_ID']
                rel_q=samples[j+1]
                label=samples[j]['RELQ_RELEVANCE2ORGQ']
                target=0
                if label=='PerfectMatch':
                    target=2
                elif label=='Relevant':
                    target=1

                label='false' if label=='Irrelevant' else 'true'

                X.append(self.getFeature(ori_q,rel_q))
                _y=np.zeros((1,3),dtype=np.float32)
                _y[0,target]=1.0
                y.append(_y)
                meta.append([ori_q_id,rel_q_id,label])
                c+=1
            pbar.update(i)
        return X,y,meta


    def loadData(self):
        reader = Reader()
        print('loading data')
        self.X_train, self.y_train, self.meta_train=self.prepareData(reader.getData(TRAIN))
        print('train data has been loaded!')
        self.X_valid, self.y_valid, self.meta_valid=self.prepareData(reader.getData(DEV))
        print('valid data has been loaded!')
        self.X_test, self.y_test, self.meta_test=self.prepareData(reader.getData(TEST))
        print('test data has been loaded!')


    def evaluate(self):
        print('evaluating...')


        f=open('./tmp/dev.pred', 'w')
        for j in range(len(self.X_valid_batch)):
                y_pred = self.predict(self.X_valid_batch[j])
                for k in range(len(y_pred)):
                    idx = j*self.batch_size+k
                    if idx>=len(self.X_valid): break
                    prob_of_true =y_pred[k,1]+y_pred[k,2]
                    label='false'
                    if prob_of_true>0.5:
                        label='true'
                    f.write( "%s %s 0 %20.16f %s\n" %(self.meta_valid[idx][0], self.meta_valid[idx][1], prob_of_true, label))
        f.close()

        map=eval_reranker(res_fname='./data/eval/SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy',
                          pred_fname='./tmp/dev.pred')
        f=open('valid_map.txt', 'a')
        f.write(str(map)+'\n')
        f.close()
        print('=========================================')
        return map


    def train(self):

        f=open('valid_map.txt', 'w')
        f.close()
        f_train_loss=open('./train_loss.txt','w')
        f_valid_loss=open('./valid_loss.txt','w')
        f_train_acc=open('./train_acc.txt','w')
        f_valid_acc=open('./valid_acc.txt','w')
        f_train_loss.close()
        f_valid_loss.close()
        f_train_acc.close()
        f_valid_acc.close()

        init_op = tf.initialize_all_variables()
        self.session.run(init_op)

        print('preparing train batch data')
        X_batch=[]
        y_batch=[]
        for j in range(1+(len(self.X_train)/self.batch_size)):
                _X=[None]*4
                for k in range(4):
                    _X[k] = [self.X_train[p][k] for p in range(j*self.batch_size,min(len(self.X_train),(j+1)*self.batch_size))]
                    _X[k] = np.concatenate(tuple(_X[k]), axis=1)
                    _size = _X[k].shape[1]
                    if _size<self.batch_size:
                        num_len = _X[k].shape[0]
                        _X[k] = np.concatenate((_X[k], np.zeros((num_len,self.batch_size-_size,self.w2v_dim),dtype=np.float32)),axis=1)

                _y = np.concatenate(tuple(self.y_train[j*self.batch_size:min(len(self.X_train),(j+1)*self.batch_size)]), axis=0)
                _size = _y.shape[0]
                if _size<self.batch_size:
                    _y = np.concatenate((_y,np.zeros((self.batch_size-_size,3),dtype=np.float32)))
                X_batch.append(_X)
                y_batch.append(_y)

        print('preparing valid batch data')
        self.X_valid_batch=[]
        for j in range(1+(len(self.X_valid)/self.batch_size)):
                _X=[None]*4
                for k in range(4):
                    _X[k] = [self.X_valid[p][k] for p in range(j*self.batch_size,min(len(self.X_valid),(j+1)*self.batch_size))]
                    _X[k] = np.concatenate(tuple(_X[k]), axis=1)
                    _size = _X[k].shape[1]
                    if _size<self.batch_size:
                        num_len = _X[k].shape[0]
                        _X[k] = np.concatenate((_X[k], np.zeros((num_len,self.batch_size-_size,self.w2v_dim),dtype=np.float32)),axis=1)
                self.X_valid_batch.append(_X)

        print("Training...")
        max_map=0.0
        for i in range(self.num_epoch):

            pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(X_batch)).start()
            loss=0.0
            for j in range(len(X_batch)):
                _loss=self.fit(X_batch[j], y_batch[j])
                loss=loss+_loss
                pbar.update(j)

            loss=loss/len(self.X_train)



            f_train_loss=open('./train_loss.txt','a')
            f_train_loss.write(str(loss)+'\n')
            f_train_loss.close()

            map=self.evaluate()
            print('MAP on valid data: %16.16f\n'%(map))
            print('train loss: %16.16f\n'%(loss))
            if map>max_map:
                max_map=map
                #self.model.save_weights("./tmp/weights.hdf5")


        print('Training completed!')

model=Model()
model.loadData()
model.train()



