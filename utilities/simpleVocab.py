from progressbar import Bar, ETA, Percentage, ProgressBar, RotatingMarker
import jieba
import pickle
from pymongo import MongoClient

class SimpleVocab:
    def __init__(self):
        self.word_to_idx={}
        self.words=[]
        self.vocabulary_size=10000
        self.reserved=['PAD','UNK_0','UNK_1','UNK_2','UNK_3']

    def add_word(self, word):
        if word not in self.word_to_idx:
            self.word_to_idx[word]=len(self.words)
            self.words.append({'word':word,'freq':1})
        else:
            self.words[self.word_to_idx[word]]['freq']+=1

    def add_reserved_words(self):
        for word in self.reserved:
            self.add_word(word)

    def scan(self, storage):
        self.add_reserved_words()
        print('Fitting storage...')
        widgets = ['Progress: ', Percentage(), ' ', Bar(),' ', ETA()]

        # Step #1 find a list of most 'vocabulary_size' frequent words
        cursor=storage.find()
        num_session=cursor.count()
        count_session=0
        pbar = ProgressBar(widgets=widgets, maxval=num_session).start()

        for message in cursor:
            for word in " ".join(jieba.cut(message['content'])).split():
                self.add_word(word)
            count_session+=1
            pbar.update(count_session)
        pbar.finish()

        print('Sorting vocab...')
        self.words[len(self.reserved):]=sorted(self.words[len(self.reserved):],key=lambda item:item['freq'],reverse=True)
        self.vocabulary_size=min(self.vocabulary_size,len(self.words))
        self.words=self.words[0:self.vocabulary_size]

        self.word_to_idx.clear()
        for i in range(len(self.words)):
            self.word_to_idx[self.words[i]['word']]=i

        print(self.words[0:100])
        print('Done!')

    def save(self,fname):
        pickle.dump([self.word_to_idx,self.words],open( fname, "wb" ))

    def load(self,fname):
        l=pickle.load(open(fname, "rb" ))
        self.word_to_idx=l[0]
        self.words=l[1]

    def to_idx(self, word):
        if word in self.word_to_idx:
            return self.word_to_idx[word]
        return None

    def to_word(self, idx):
        if idx<len(self.words):
            return self.words[idx]
        return None

'''
Tests

db = MongoClient().sonny['dataset']
sv = SimpleVocab()
sv.scan(db)
sv.save('/root/source/Sonny/experiments/data/simple_vocab.dat')
'''