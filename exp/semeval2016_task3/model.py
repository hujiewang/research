import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
import numpy as np
from experiments.tfidf.SparseMatrixStorage import SparseMatrixStorage
from experiments.tfidf.sparseDotProduct import sparseDot
from pymongo import MongoClient
import math
import random
import jieba
from gensim.models.word2vec import Word2Vec
from experiments.feature import BM25
import pickle
from reader import Reader,TRAIN,TEST,DEV
from ev2 import eval_reranker
from preprocess import preprocess
from gensim.models.phrases import Phrases

from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    AdaptiveETA, FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer

# For reproducibility
random.seed()

# Average document length
AVGDL = 15.66

# Document number
DOCNUM = 344177

# Number of Features
BASIC_FEATURE_NUM = 0
# TERM_FEATURE_NUM = 1
# TOPIC_FEATURE_NUM = LSA_TOPIC_NUM
TERM_FEATURE_NUM = 0
TOPIC_FEATURE_NUM = 0

WORDVECTOR_FEATURE_NUM = 100
FEATURE_NUM = BASIC_FEATURE_NUM + TERM_FEATURE_NUM + TOPIC_FEATURE_NUM + 2*WORDVECTOR_FEATURE_NUM



class defaultConfig(object):
    dropout_prob = 0.5
    num_features = 400
    num_hidden_1 = 200
    num_hidden_2 =1
    max_grad_norm = 5

class RankNet:
    def __init__(self, is_training=True, config=defaultConfig):

        self.config=config

        def getModel(input):
            # 2-layer NN
            with tf.variable_scope("NN", initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0)):
                W_1 = tf.get_variable("W_1", [self.config.num_hidden_1, input.get_shape()[0]])
                self._test=W_1
                b_1 = tf.get_variable("b_1", [self.config.num_hidden_1,1])
                W_2 = tf.get_variable("W_2", [self.config.num_hidden_2, self.config.num_hidden_1])
                b_2 = tf.get_variable("b_2", [self.config.num_hidden_2,1])
                y_1 = tf.sigmoid(tf.matmul(W_1, input)+b_1)
                y_2 = tf.sigmoid(tf.matmul(W_2, y_1)+b_2)
            return y_2

        self._input_0 = tf.placeholder(tf.float32, [self.config.num_features,1], name='_input_0')
        self._input_1 = tf.placeholder(tf.float32, [self.config.num_features,1], name='_input_1')
        self._target = tf.placeholder(tf.float32)

        with tf.variable_scope("Merge") as scope:
            o_i=getModel(self._input_0)
            scope.reuse_variables()
            o_j=getModel(self._input_1)
            self._score = o_i
        o_ij=o_i-o_j
        self._cost=-self._target*o_ij+tf.log(1+tf.exp(o_ij))
        self._lr = tf.Variable(0.0, trainable=False)
        self._train_op = tf.train.AdamOptimizer().minimize(self._cost)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class Trainer:
    def __init__(self,train_data,dev_data,test_data):
        self.train_data=train_data
        self.dev_data=dev_data
        self.test_data=test_data

        # Hyper-parameters
        self.learningRate=0.01
        self.trainSize=2000
        self.testSize=1000
        self.totalSize = self.trainSize + self.testSize
        self.maxEpochs=10000
        self.num_processed=-1

        self.w2v_model=Word2Vec.load('./data/word2vec/w2v.model')
        self.bigram=Phrases.load('./data/bigram.dat')
        self.trigram=Phrases.load('./data/trigram.dat')

    def fit(self, session, m, data, label):
        """Runs the model on the given data."""
        cost, _, test= session.run([m._cost, m._train_op, m._test],{m._input_0: data[0], m._input_1: data[1], m._target: label})
        return cost

    def score(self, session, m, data):
        """Runs the model on the given data."""
        score = session.run([m._score],{m._input_0: data, m._input_1: data})
        return score

    def attr2num(self,str):
        if str=='Irrelevant':
            return 0
        elif str=='Relevant':
            return 1
        elif str=='PerfectMatch':
            return 2
        else:
            print('invalid attri!!')

    def generateTmpTrain(self, cur_ori_q_idx):
        '''
        data: train or dev
        '''

        q_data=self.train_data[cur_ori_q_idx][1]
        tmp_train=[]
        _len = int((len(self.train_data[cur_ori_q_idx])-2)/2)
        for i in range(_len):
            for j in range(_len):
                idx_i=(i+1)*2
                idx_j=(j+1)*2
                attr_i=self.train_data[cur_ori_q_idx][idx_i]
                attr_j=self.train_data[cur_ori_q_idx][idx_j]
                label=None
                num_attr_i=self.attr2num(attr_i['RELQ_RELEVANCE2ORGQ'])
                num_attr_j=self.attr2num(attr_j['RELQ_RELEVANCE2ORGQ'])
                if num_attr_i<num_attr_j:
                    label=0
                elif num_attr_i>num_attr_j:
                    label=1
                else:
                    continue
                tmp_train.append([self.train_data[cur_ori_q_idx][idx_i+1],self.train_data[cur_ori_q_idx][idx_j+1],label])
        return q_data, tmp_train


    def getData(self):
        if self.cur_idx is None or self.cur_idx==len(self.tmp_train):
            if self.cur_ori_q_idx is None:
                self.cur_ori_q_idx=0
            else:
                self.cur_ori_q_idx+=1
            if self.cur_ori_q_idx == len(self.train_data):
                return None, None
            self.tmp_train=[]
            while True:
                self.tmp_q, self.tmp_train=self.generateTmpTrain(self.cur_ori_q_idx)
                if len(self.tmp_train)>0:
                    break
                else:
                    self.cur_ori_q_idx+=1
                    if self.cur_ori_q_idx == len(self.train_data):
                        return None, None
            self.cur_idx=0

        curData=self.tmp_train[self.cur_idx]
        self.cur_idx+=1

        feature_0=self.getFeature(self.tmp_q,curData[0])
        feature_1=self.getFeature(self.tmp_q,curData[1])

        return [feature_0,feature_1], curData[2]


    '''
    word2vec related stuff
    '''
    def getWordVectorFeatures(self, text):
        words = text.split()
        return self.wordVectorAvg(words, self.w2v_model, WORDVECTOR_FEATURE_NUM)

    def wordVectorAvg(self, words, model, num_features):
        featureVec = np.zeros((num_features,1),dtype="float32")

        nwords = 0

        # Index2word is a list that contains the names of the words in
        # the model's vocabulary. Convert it to a set, for speed
        index2word_set = set(model.index2word)

        # Loop over each word in the review and, if it is in the model's
        # vocaublary, add its feature vector to the total
        for word in words:
            if word in index2word_set:
                nwords = nwords + 1
                featureVec = np.add(featureVec, model[word].reshape(-1,1))

        # Divide the result by the number of words to get the average
        if nwords!=0:
            featureVec = np.divide(featureVec, nwords)
        return featureVec

    def getFeature(self, ori_q,rel_q):
        ori_q[0]=preprocess(ori_q[0],no_stopwords=True,bigram=self.bigram,trigram=self.trigram)
        ori_q[1]=preprocess(ori_q[1],no_stopwords=True,bigram=self.bigram,trigram=self.trigram)

        rel_q[0]=preprocess(rel_q[0],no_stopwords=True,bigram=self.bigram,trigram=self.trigram)
        rel_q[1]=preprocess(rel_q[1],no_stopwords=True,bigram=self.bigram,trigram=self.trigram)

        word2vec_q_subject=self.getWordVectorFeatures(ori_q[0])
        word2vec_q_body=self.getWordVectorFeatures(ori_q[1])
        word2vec_q=np.concatenate((word2vec_q_subject,word2vec_q_body),axis=0)

        word2vec_rel_q_subject=self.getWordVectorFeatures(rel_q[0])
        word2vec_rel_q_body=self.getWordVectorFeatures(rel_q[1])
        word2vec_rel_q=np.concatenate((word2vec_rel_q_subject,word2vec_rel_q_body),axis=0)

        return np.concatenate((word2vec_q,word2vec_rel_q),axis=0)

    def predict(self, session, mvalid, fname, data):

        print('predicting....')
        f=open(fname, 'w')

        for i in range(len(data)):
            samples = data[i]

            ori_q_id=samples[0]['ORGQ_ID']
            ori_q=samples[1]

            for j in range(2,len(samples),2):

                rel_q_id=samples[j]['RELQ_ID']
                rel_q=samples[j+1]
                label=samples[j]['RELQ_RELEVANCE2ORGQ']
                if label=='PerfectMatch' or label=='Relevant':
                    label='true'
                elif label=='Irrelevant':
                    label='false'
                elif label=='?':
                    # this is a test data
                    label='true'

                feature = self.getFeature(ori_q,rel_q)
                score=self.score(session, mvalid, feature)
                s=score[0][0,0]
                f.write(ori_q_id+' '+rel_q_id+' 0 '+str(s)+' '+label+'\n')
        f.close()
        print('predictions have been generated!')

    def eval(self, fname):

        print('evaluating...')
        map=eval_reranker(res_fname='./data/eval/SemEval2016-Task3-CQA-QL-dev.xml.subtaskB.relevancy',
                          pred_fname=fname)
        f=open('valid_map.txt', 'a')
        f.write(str(map)+'\n')
        f.close()
        print('=========================================')

    def train(self):
        session = tf.Session()

        with tf.variable_scope("model", reuse=None):
            m = RankNet(is_training=True)
        with tf.variable_scope("model", reuse=True):
            mvalid = RankNet(is_training=False)
            mtest = RankNet(is_training=False)

        init_op = tf.initialize_all_variables()
        m.assign_lr(session, self.learningRate)
        session.run(init_op)

        f=open('train_loss.txt','w')
        f.close()
        f=open('valid_map.txt','w')
        f.close()
        for i in range(self.maxEpochs):

            # Training

            total_loss=0.0
            self.cur_idx=None
            self.cur_ori_q_idx=None

            pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(self.train_data)).start()

            self.trainSize=0

            while True:
                data, label = self.getData()
                if data is None:
                    break

                total_loss += self.fit(session, m,data, label)
                self.trainSize+=1
                pbar.update(self.cur_ori_q_idx)
            pbar.finish()
            print('Number of Samples: '+str(self.trainSize))
            print(str(total_loss[0,0]/self.trainSize)+'\n')
            f=open('train_loss.txt', 'a')
            f.write(str(total_loss[0,0]/self.trainSize)+'\n')
            f.close()


            # cross validation
            self.predict(session, mvalid, './tmp/dev.pred',self.dev_data)
            self.eval('./tmp/dev.pred')

        session.close()




reader = Reader()
train_data=reader.getData(TRAIN)
dev_data=reader.getData(DEV)
test_data=reader.getData(TEST)
trainer = Trainer(train_data,dev_data,test_data)
trainer.train()




