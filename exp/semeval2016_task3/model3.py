from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from reader import Reader,TRAIN,TEST,DEV
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from ev2 import eval_reranker
from preprocess import preprocess
from gensim.models.word2vec import Word2Vec
from sklearn.utils import shuffle
from keras.regularizers import l2
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from gensim.models.phrases import Phrases
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    AdaptiveETA, FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer

class Model(object):
    def __init__(self):

        '''
        Training parameters:
        '''

        self.w2v_dim=100
        self.num_feature=400
        self.batch_size=16
        self.num_epoch=1

        #self.w2v_model=Word2Vec.load_word2vec_format('./data/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
        self.w2v_model=Word2Vec.load('./data/word2vec/w2v.model')
        self.index2word_set = set(self.w2v_model.index2word)
        self.bigram=Phrases.load('./data/bigram.dat')
        self.trigram=Phrases.load('./data/trigram.dat')

        print('Build model...')

        param_dist = {
            "n_estimators":sp_randint(20,250),
            "criterion": ["gini", "entropy"],
            "max_depth": sp_randint(10, 300),
            "min_samples_split": sp_randint(1, 30),
            "min_samples_leaf": sp_randint(1, 30),
            "max_features": sp_randint(1, 200),
            "bootstrap": [True, False],
            'random_state':sp_randint(1, 1000000),
        }
        # build a classifier
        clf = RandomForestClassifier(n_jobs=8)
        # run randomized search
        self.model=RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=10,cv=9,n_jobs=8)

        print('Model has been built!')

    def getWordVectorFeatures(self, text):
        words = text.split()
        return self.wordVectorAvg(words, self.w2v_dim)

    def wordVectorAvg(self, words, num_features):
        featureVec = np.zeros((num_features,1),dtype="float32")

        nwords = 0
        for word in words:
            if word in self.index2word_set:
                nwords = nwords + 1
                featureVec = np.add(featureVec, self.w2v_model[word].reshape(-1,1))

        if nwords!=0:
            featureVec = np.divide(featureVec, nwords)
        return featureVec

    def getFeature(self, ori_q,rel_q):
        ori_q[0]=preprocess(ori_q[0],no_stopwords=True,bigram=self.bigram,trigram=self.trigram)
        ori_q[1]=preprocess(ori_q[1],no_stopwords=True,bigram=self.bigram,trigram=self.trigram)

        rel_q[0]=preprocess(rel_q[0],no_stopwords=True,bigram=self.bigram,trigram=self.trigram)
        rel_q[0]=preprocess(rel_q[0],no_stopwords=True,bigram=self.bigram,trigram=self.trigram)

        word2vec_q_subject=self.getWordVectorFeatures(ori_q[0])
        word2vec_q_body=self.getWordVectorFeatures(ori_q[1])

        word2vec_rel_q_subject=self.getWordVectorFeatures(rel_q[0])
        word2vec_rel_q_body=self.getWordVectorFeatures(rel_q[1])



        subject=np.concatenate((word2vec_q_subject*word2vec_rel_q_subject,
                                np.abs(word2vec_q_subject-word2vec_rel_q_subject)),axis=0)

        body=np.concatenate((word2vec_q_body*word2vec_rel_q_body,
                                np.abs(word2vec_q_body-word2vec_rel_q_body)),axis=0)


        return np.concatenate((subject, body,),axis=0).T


    def prepareData(self,data):
        size=0
        for i in range(len(data)):
            size+=(len(data[i])/2)-1
        X=np.zeros((size,self.num_feature),dtype=np.float32)
        y=np.zeros((size,),dtype=np.float32)
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

                X[c,:] = self.getFeature(ori_q,rel_q)
                y[c]=target
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

        y_pred = self.model.predict_proba(self.X_valid)
        f=open('./tmp/dev.pred', 'w')
        for i in range(len(self.meta_valid)):
            prob_of_true =y_pred[i][1]+y_pred[i][2]
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

        print("Training...")
        max_map=0.0
        for i in range(self.num_epoch):
            self.model.fit(self.X_train, self.y_train)

            '''
            f_train_loss=open('./train_loss.txt','a')
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
            if map>max_map:
                max_map=map
                #self.model.save_weights("./tmp/weights.hdf5")


        print('Training completed!')

model=Model()
model.loadData()
model.train()



