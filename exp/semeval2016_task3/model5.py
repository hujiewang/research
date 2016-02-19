
from reader import Reader,TRAIN,TEST,DEV
import numpy as np
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
from tensorflow.python.ops.rnn import rnn
from tensorflow.python.ops import variable_scope as vs
import numpy as np

class Model(object):
    def __init__(self):

        '''
        Training parameters:
        '''

        self.w2v_dim=100
        self.num_feature=400
        self.batch_size=16
        self.num_epoch=100
        self.num_hidden_1=64
        self.num_hidden_2=3

        self.max_len=10

        #self.w2v_model=Word2Vec.load_word2vec_format('./data/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
        self.w2v_model=Word2Vec.load('./data/word2vec/w2v.model')
        self.index2word_set = set(self.w2v_model.index2word)
        self.bigram=Phrases.load('./data/bigram.dat')
        self.trigram=Phrases.load('./data/trigram.dat')

        # Model
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_0=tf.placeholder(tf.float32,[self.max_len,1,self.w2v_dim])
            self.input_1=tf.placeholder(tf.float32,[self.max_len,1,self.w2v_dim])
            self.length_0=tf.placeholder(tf.int32)
            self.length_1=tf.placeholder(tf.int32)
            self.target = tf.placeholder(tf.float32, [3])

            input_0=array_ops.unpack(self.input_0)
            input_1=array_ops.unpack(self.input_1)

            cell=rnn_cell.GRUCell(self.w2v_dim)
            with vs.variable_scope('RNN'):
                output_0, state_0 = rnn(cell, input_0,  dtype=tf.float32)
                tf.get_variable_scope().reuse_variables()
                output_1, state_1 = rnn(cell, input_1,  dtype=tf.float32)

            state=tf.concat(1,[state_0[-1],state_1[-1]])
             # 2-layer NN
            with tf.variable_scope("NN", initializer=tf.random_uniform_initializer()):
                    W_1 = tf.get_variable("W_1", [state.get_shape()[1],self.num_hidden_1])
                    b_1 = tf.get_variable("b_1", [self.num_hidden_1])
                    W_2 = tf.get_variable("W_2", [self.num_hidden_1,self.num_hidden_2])
                    b_2 = tf.get_variable("b_2", [self.num_hidden_2])
                    tmp = tf.matmul(state, W_1)
                    y_1 = tf.sigmoid(tf.matmul(state, W_1)+b_1)
                    y_pred = tf.sigmoid(tf.matmul(y_1, W_2)+b_2)

            self.y_pred=tf.nn.softmax(y_pred)
            self.cross_entropy = -tf.reduce_sum(self.target*tf.log(self.y_pred))
            #self.train_step = tf.train.AdamOptimizer().minimize(self.cross_entropy)
            # Optimizer.
            global_step = tf.Variable(0)
            optimizer = tf.train.GradientDescentOptimizer(0.1)
            gradients, v = zip(*optimizer.compute_gradients(self.cross_entropy))
            gradients, _ = tf.clip_by_global_norm(gradients, 5)
            self.optimizer= optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

        print('Model has been built!')

    def fit(self, X, y):
        _, loss =self.session.run([self.optimizer, self.cross_entropy],{self.input_0: X[0],
                                                                        self.input_1: X[1],
                                                                        self.length_0:X[2],
                                                                        self.length_1:X[3],
                                                                        self.target: y})
        return loss

    def predict(self, X):
        y_pred =self.session.run([self.y_pred],{self.input_0: X[0],self.input_1: X[1]})
        return y_pred

    def getWordVectorFeatures(self, text):
        words = text.split()

        featureVec=np.zeros((self.max_len,1,self.w2v_dim))
        for i in range(len(words)):
            if i==self.max_len:break
            word=words[i]
            if word in self.index2word_set:
                featureVec[i,0,:]=self.w2v_model[word]
        return featureVec, len(words)


    def getFeature(self, ori_q,rel_q):
        ori_q[0]=preprocess(ori_q[0],bigram=self.bigram,trigram=self.trigram)
        rel_q[0]=preprocess(rel_q[0],bigram=self.bigram,trigram=self.trigram)

        word2vec_q_subject, q_len=self.getWordVectorFeatures(ori_q[0])
        word2vec_rel_q_subject, rel_q_len=self.getWordVectorFeatures(rel_q[0])

        return [word2vec_q_subject, word2vec_rel_q_subject, q_len, rel_q_len]


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
                _y=np.zeros((3,),dtype=np.float32)
                _y[target]=1.0
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
        for i in range(len(self.meta_valid)):
            y_pred = self.predict(self.X_valid[i])
            prob_of_true =y_pred[0][0,1]+y_pred[0][0,2]
            label='false'
            if prob_of_true>0.5:
                label='true'
            f.write( "%s %s 0 %20.16f %s\n" %(self.meta_valid[i][0], self.meta_valid[i][1], prob_of_true, label))
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

        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph)
            tf.initialize_all_variables().run(session=self.session)

        print("Training...")
        max_map=0.0
        for i in range(self.num_epoch):

            pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(self.X_train)).start()
            loss=0.0
            for j in range(len(self.X_train)):
                _loss=self.fit(self.X_train[i], self.y_train[i])
                loss=loss+_loss
                pbar.update(j)

            loss=loss/len(self.X_train)



            f_train_loss=open('./train_loss.txt','a')
            f_train_loss.write(str(loss)+'\n')
            f_train_loss.close()
            '''
            f_valid_loss=open('./valid_loss.txt','a')
            f_train_acc=open('./train_acc.txt','a')
            f_valid_acc=open('./valid_acc.txt','a')

            f_train_loss.write(str(np.asscalar(hist.history['loss'][0])))
            f_train_loss.write('\n')
            f_valid_loss.write(str(np.asscalar(hist.history['val_loss'][0])))
            f_valid_loss.write('\n')

            f_train_acc.write(str(np.asscalar(hist.history['acc'][0])))
            f_train_acc.write('\n')
            f_valid_acc.write(str(np.asscalar(hist.history['val_acc'][0])))
            f_valid_acc.write('\n')

            f_train_loss.close()
            f_valid_loss.close()
            f_train_acc.close()
            f_valid_acc.close()
            '''

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



