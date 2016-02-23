
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    AdaptiveETA, FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer
import os
from nltk.tokenize import RegexpTokenizer
import re
import pickle
import math
from sklearn.cross_validation import train_test_split
class Model:
    def __init__(self):

        self.max_len = 300
        self.seg_len = 5
        self.batch_size = 100
        self.number_of_layers = 1
        self.dim = 59+2
        self.num_epoch=10000
        self.num_hidden_1=2
        self.num_hidden_2=2

        self.input=tf.placeholder(tf.float32,[self.max_len,None,self.dim])
        self.target = tf.placeholder(tf.float32, [None, 2])
        self.keep_prob = tf.placeholder(tf.float32)
        input=array_ops.unpack(self.input)
        batch_size = tf.shape(self.input)[1]

        def _rnn(cell, inputs):
            with tf.variable_scope("GRU_RNN") as scope:
                state = cell.zero_state(batch_size, tf.float32)
                for time, input_ in enumerate(inputs):
                    if time > 0: scope.reuse_variables()
                    output, state = cell(input_, state)
                return state

        def h_rnn(input):
            i=0
            num_layer=0
            layer=[input]
            while True:
                print(num_layer)
                layer.append([])
                _input=layer[num_layer]
                length = len(_input)
                with tf.variable_scope("RNN_"+str(num_layer)) as scope:
                    cell=rnn_cell.BasicLSTMCell(self.dim)
                    cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                    stacked_cell = rnn_cell.MultiRNNCell([cell] * self.number_of_layers)
                    i=0
                    while i<length:
                        state = _rnn(stacked_cell, _input[i:min(i+self.seg_len,length)])
                        layer[num_layer+1].append(state)
                        scope.reuse_variables()
                        i+=self.seg_len
                num_layer+=1
                if length<=self.seg_len:
                    break

            return layer[num_layer][0]



        state = h_rnn(input)

        with tf.variable_scope("NN", initializer=tf.random_uniform_initializer()):
            self.W_1 = tf.get_variable("W_1", [state.get_shape()[1],self.num_hidden_1])
            self.b_1 = tf.get_variable("b_1", [self.num_hidden_1])
            # self.W_2 = tf.get_variable("W_2", [self.num_hidden_1,self.num_hidden_2])
            # self.b_2 = tf.get_variable("b_2", [self.num_hidden_2])

        y_1 = tf.matmul(state, self.W_1)+self.b_1
        # y_1 = tf.nn.sigmoid(tf.matmul(state, self.W_1)+self.b_1)
        # y_2 = tf.matmul(y_1, self.W_2)+self.b_2
        self.y_pred = tf.nn.softmax(y_1)
        self.cross_entropy = -tf.reduce_mean(self.target*tf.log(self.y_pred))

        correct_prediction = tf.equal(tf.argmax(self.target, 1), tf.argmax(self.y_pred, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_summary = tf.scalar_summary("accuracy", self.accuracy)
        ce_summ = tf.scalar_summary("cross entropy", self.cross_entropy)
        self.merged = tf.merge_all_summaries()


        # Optimizer.
        global_step = tf.Variable(0)
        # optimizer = tf.train.GradientDescentOptimizer(0.1)
        optimizer = tf.train.AdamOptimizer(0.01)
        gradients, v = zip(*optimizer.compute_gradients(self.cross_entropy))
        gradients, _ = tf.clip_by_global_norm(gradients, 10)
        self.optimizer= optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

    def test_mnist(self):
        x=self.mnist.test.images
        _y = self.mnist.test.labels
        _x = []
        for j in range(len(x)):
            _x.append(x[j,:].reshape(28,1,28))
        _x=np.concatenate(tuple(_x), axis=1)
        acc = self.session.run(self.accuracy, feed_dict={self.input: _x, self.target: _y, self.keep_prob: 1.0})
        f_train_loss=open('./valid_acc.txt','a')
        f_train_loss.write(str(acc)+'\n')
        f_train_loss.close()
        print('accuracy: {}\n'.format(acc))

    def train_mnist(self):
        f_train_loss=open('./train_loss.txt','w')
        f_valid_acc=open('./valid_acc.txt','w')
        f_train_loss.close()
        f_valid_acc.close()

        self.session = tf.Session()
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        writer = tf.train.SummaryWriter("/tmp/mnist_logs", self.session.graph_def)
        init_op = tf.initialize_all_variables()
        self.session.run(init_op)

        X=[]
        y=[]
        for i in range(1000):
                x, _y = self.mnist.train.next_batch(self.batch_size)
                _x = []
                for j in range(len(x)):
                    _x.append(x[j,:].reshape(28,1,28))
                _x=np.concatenate(tuple(_x), axis=1)
                X.append(_x)
                y.append(_y)

        for epoch in range(self.num_epoch):
            pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=1000).start()
            loss=0.0
            for i in range(1000):
                results = self.session.run([self.merged,self.cross_entropy,self.optimizer], feed_dict={self.input: X[i], self.target: y[i], self.keep_prob: 0.5})
                writer.add_summary(results[0], i)
                loss+=results[1]
                pbar.update(i+1)
            print('\nloss: {}\n'.format(results[1]))
            f_train_loss=open('./train_loss.txt','a')
            f_train_loss.write(str(loss/1000)+'\n')
            f_train_loss.close()
            self.test()
        # Test trained model


    def preprocess(self, text):
        # tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
        text=text.rstrip('\n')
        text=text.rstrip('\t')
        text=text.rstrip('\n\t')
        text=text.rstrip('\n\t')
        text=re.sub("<br />",'', text)
        text=text.lower()
        # text=' '.join(tokenizer.tokenize(text))
        # text=re.sub("[0-9]+",'NUMBER',text)
        x = np.zeros((self.max_len, 1, self.dim),dtype=np.float32)
        for i in range(self.max_len):
            if i >=len(text):
                x[i,:,:]=np.zeros((1,self.dim))
                x[i,0,0]=1.0
                continue
            ch = text[i]
            x[i,:,:]=np.zeros((1,self.dim))
            idx=1
            if ch in self.dict:
                idx=self.dict[ch]
            x[i,0,idx]=1.0


        return x

    def prepareData(self):
        X=[]
        y=[]
        chars='abcdefghiklmnopqrstuvwxyz0123456789!@#$%^&*{}[],.()<>?/\\:\";'""

        # self.dict = pickle.load(open('./dict.dat','r'))
        self.dict={}
        self.dict['UNK']=1
        self.dict['PAD']=0
        for i in range(len(chars)):
            self.dict[chars[i]]=i+2
        '''
        dict_size=0
        self.dict={}
        for file in os.listdir("data/aclImdb/train/pos"):
            f=open('data/aclImdb/train/pos/'+file,'r')
            line = f.readline()
            for ch in line:
                if ch not in self.dict:
                    self.dict[ch]=dict_size
                    dict_size+=1
            # X.append(self.preprocess(line))
            # y.append(np.array([0,1]))
        for file in os.listdir("data/aclImdb/train/neg"):
            f=open('data/aclImdb/train/neg/'+file,'r')
            line = f.readline()
            for ch in line:
                if ch not in self.dict:
                    self.dict[ch]=dict_size
                    dict_size+=1
            # X.append(self.preprocess(line))
            # y.append(np.array([1,0]))
        f=open('./dict.dat','w')
        pickle.dump(self.dict,f)
        f.close()
        '''
        for file in os.listdir("data/aclImdb/train/pos"):
            f=open('data/aclImdb/train/pos/'+file,'r')
            line = f.readline()
            X.append(self.preprocess(line))
            y.append(np.array([1,0]).reshape((1,-1)))
        for file in os.listdir("data/aclImdb/train/neg"):
            f=open('data/aclImdb/train/neg/'+file,'r')
            line = f.readline()
            X.append(self.preprocess(line))
            y.append(np.array([0,1]).reshape((1,-1)))

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

        print('preparing train batch data')
        self.X_batch=[]
        self.y_batch=[]
        for j in range(int(math.ceil(len(X_train)/float(self.batch_size)))):
                _X=np.concatenate(tuple(X_train[j*self.batch_size:min(len(X_train),(j+1)*self.batch_size)]), axis=1)
                _y = np.concatenate(tuple(y_train[j*self.batch_size:min(len(X_train),(j+1)*self.batch_size)]), axis=0)
                self.X_batch.append(_X)
                self.y_batch.append(_y)

        print('preparing valid batch data')
        self.X_valid_batch=[]
        self.y_valid_batch=[]
        for j in range(int(math.ceil(len(X_valid)/float(self.batch_size)))):
                _X=np.concatenate(tuple(X_valid[j*self.batch_size:min(len(X_valid),(j+1)*self.batch_size)]), axis=1)
                _y = np.concatenate(tuple(y_valid[j*self.batch_size:min(len(X_valid),(j+1)*self.batch_size)]), axis=0)
                self.X_valid_batch.append(_X)
                self.y_valid_batch.append(_y)

    def train_imdb(self):
        f_train_loss=open('./train_loss.txt','w')
        f_valid_acc=open('./valid_acc.txt','w')
        f_train_loss.close()
        f_valid_acc.close()

        self.session = tf.Session()
        init_op = tf.initialize_all_variables()
        self.session.run(init_op)

        print("Training...")
        for epoch in range(self.num_epoch):
            pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(self.X_batch)).start()
            loss=0.0
            for i in range(len(self.X_batch)):
                results = self.session.run([self.cross_entropy,self.optimizer], feed_dict={self.input: self.X_batch[i], self.target: self.y_batch[i], self.keep_prob: 0.5})
                loss+=results[0]
                pbar.update(i+1)
            print('\nloss: {}\n'.format(loss/len(self.X_batch)))
            f_train_loss=open('./train_loss.txt','a')
            f_train_loss.write(str(loss/len(self.X_batch))+'\n')
            f_train_loss.close()

            for i in range(len(self.X_valid_batch)):
                acc = self.session.run(self.accuracy, feed_dict={self.input:self.X_valid_batch[i], self.target: self.y_valid_batch[i], self.keep_prob: 1.0})
            f_train_loss=open('./valid_acc.txt','a')
            f_train_loss.write(str(acc/len(self.X_valid_batch))+'\n')
            f_train_loss.close()
            print('\naccuracy: {}\n'.format(acc/len(self.X_valid_batch)))

        print('Training completed!')





m=Model()
m.prepareData()
m.train_imdb()








