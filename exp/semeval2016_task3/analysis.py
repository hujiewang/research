from reader import Reader,TRAIN,TEST,DEV
import matplotlib.pyplot as plt
from preprocess import preprocess
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phrases
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    AdaptiveETA, FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer
from scipy import stats

class Anaysis:
    def __init__(self):
        reader = Reader()
        print('loading data')
        self.X_train=reader.getData(TRAIN)
        print('train data has been loaded!')
        self.X_valid=reader.getData(DEV)
        print('valid data has been loaded!')
        self.X_test=reader.getData(TEST)
        print('test data has been loaded!')
        self.c_title=[]
        self.c_body=[]
        self.bigram=Phrases.load('./data/bigram.dat')
        self.trigram=Phrases.load('./data/trigram.dat')

    def count(self, ori_q, rel_q):
        ori_q[0]=preprocess(ori_q[0],bigram=self.bigram,trigram=self.trigram)
        rel_q[0]=preprocess(rel_q[0],bigram=self.bigram,trigram=self.trigram)
        ori_q[1]=preprocess(ori_q[1],bigram=self.bigram,trigram=self.trigram)
        rel_q[1]=preprocess(rel_q[1],bigram=self.bigram,trigram=self.trigram)
        self.c_title.append(len(ori_q[0].split()))
        self.c_title.append(len(rel_q[0].split()))
        self.c_body.append(len(ori_q[1].split()))
        self.c_body.append(len(rel_q[1].split()))

    def lenDistribution(self, data):
        pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(data)).start()
        for i in range(len(data)):
            samples = data[i]

            ori_q_id=samples[0]['ORGQ_ID']
            ori_q=samples[1]

            for j in range(2,len(samples),2):
                rel_q=samples[j+1]

                self.count(ori_q,rel_q)
            pbar.update(i)


    def show(self):
        print(stats.scoreatpercentile(self.c_title, 95))
        plt.hist(self.c_title)
        plt.title("Gaussian Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
        print(stats.scoreatpercentile(self.c_body, 95))
        plt.hist(self.c_body)
        plt.title("Gaussian Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()



a=Anaysis()
a.lenDistribution(a.X_train)
a.lenDistribution(a.X_valid)
a.lenDistribution(a.X_test)
a.show()
