
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
from gensim.models import Doc2Vec
import math
class Model(object):
    def __init__(self):
        self.session = tf.Session()
        '''
        Training parameters:
        '''

        self.w2v_dim=100
        self.num_feature=400
        self.batch_size=32
        self.num_epoch=10000
        self.num_hidden_1=3
        self.num_hidden_2=3

        self.number_of_layers=3

        #self.max_len = 50
        self.max_len_title=6
        self.max_len_body=38

        self.d2v_model=Doc2Vec.load('data/word2vec/d2v.model')
        #self.bigram = None
        #self.trigram =None
        self.bigram=Phrases.load('./data/bigram.dat')
        self.trigram=Phrases.load('./data/trigram.dat')

        # Model
        self.input=tf.placeholder(tf.float32,[None,self.w2v_dim*4])


        self.dropout_input = tf.placeholder(tf.float32)
        self.dropout_hidden = tf.placeholder(tf.float32)

        self.target = tf.placeholder(tf.float32, [None, 3])


         # 2-layer NN
        # 2-layer NN
        with tf.variable_scope("NN", initializer=tf.random_uniform_initializer()):
            W_1 = tf.get_variable("W_1", [self.w2v_dim*4, self.num_hidden_1])
            b_1 = tf.get_variable("b_1", [self.num_hidden_1])
            # W_2 = tf.get_variable("W_2", [self.num_hidden_1, self.num_hidden_2])
            # b_2 = tf.get_variable("b_2", [self.num_hidden_2])

            # input = tf.nn.dropout(input, self.dropout_input)
            # y_1 = tf.sigmoid(tf.matmul(self.input, W_1)+b_1)
            # y_1 = tf.nn.dropout(y_1, self.dropout_hidden)
            # y_2 = tf.matmul(y_1, W_2)+b_2
        y_2 = tf.matmul(self.input, W_1)+b_1

        self.y_pred=tf.nn.softmax(y_2)
        self.y_pred=tf.clip_by_value(self.y_pred,1e-7, 1.0)
        self.cross_entropy = -tf.reduce_mean(self.target*tf.log(self.y_pred))


        # Optimizer.

        global_step = tf.Variable(0)
        # optimizer = tf.train.GradientDescentOptimizer(0.1)
        # optimizer = tf.train.AdamOptimizer(0.01)
        # gradients, v = zip(*optimizer.compute_gradients(self.cross_entropy))
        # gradients, _ = tf.clip_by_global_norm(gradients, 50)
        # self.optimizer= optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.cross_entropy)


        print('Model has been built!')

    def fit(self, X, y):

        _, loss, y_pred=self.session.run([self.optimizer, self.cross_entropy, self.y_pred],{
                                                                        self.input: X,
                                                                        self.dropout_input: 0.2,
                                                                        self.dropout_hidden: 0.5,
                                                                        self.target: y})
        a=3
        return loss

    def predict(self, X):
        y_pred =self.session.run([self.y_pred],{self.input: X,
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
        ori_q[0]=preprocess(ori_q[0],bigram=self.bigram,trigram=self.trigram)
        rel_q[0]=preprocess(rel_q[0],bigram=self.bigram,trigram=self.trigram)
        ori_q[1]=preprocess(ori_q[1],bigram=self.bigram,trigram=self.trigram)
        rel_q[1]=preprocess(rel_q[1],bigram=self.bigram,trigram=self.trigram)

        word2vec_q_subject=self.d2v_model.infer_vector(ori_q[0].split(), steps=10)
        word2vec_rel_q_subject=self.d2v_model.infer_vector(rel_q[0].split(), steps=10)

        word2vec_q_body=self.d2v_model.infer_vector(ori_q[1].split(), steps=10)
        word2vec_rel_q_body=self.d2v_model.infer_vector(rel_q[1].split(), steps=10)

        a=np.concatenate((np.abs(word2vec_q_subject-word2vec_rel_q_subject),word2vec_q_subject*word2vec_rel_q_subject), axis=0)
        b=np.concatenate((np.abs(word2vec_q_body-word2vec_rel_q_body),word2vec_q_body*word2vec_rel_q_body), axis=0)

        return np.concatenate((a,b), axis=0).reshape((1,self.w2v_dim*4))



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
        for j in range(int(math.ceil(len(self.X_train)/float(self.batch_size)))):
                _X=np.concatenate(tuple(self.X_train[j*self.batch_size:min(len(self.X_train),(j+1)*self.batch_size)]), axis=0)
                _y = np.concatenate(tuple(self.y_train[j*self.batch_size:min(len(self.X_train),(j+1)*self.batch_size)]), axis=0)
                X_batch.append(_X)
                y_batch.append(_y)

        print('preparing valid batch data')
        self.X_valid_batch=[]
        for j in range(int(math.ceil(len(self.X_valid)/float(self.batch_size)))):
                _X=np.concatenate(tuple(self.X_valid[j*self.batch_size:min(len(self.X_valid),(j+1)*self.batch_size)]), axis=0)
                self.X_valid_batch.append(_X)

        print(len(X_batch))
        print(len(self.X_valid_batch))
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



