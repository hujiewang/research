from reader import Reader,TRAIN,TEST,DEV,EXTRA
from preprocess import preprocess
from gensim.models.phrases import Phrases
reader = Reader()
sentences=reader.getText(TRAIN+EXTRA)
# use phrase only when it has already trained


bigram=Phrases.load('./data/bigram.dat')
trigram=Phrases.load('./data/trigram.dat')
with open('./data/text_cleaned_phrase.txt','w') as f:
    for sentence in sentences:
        f.write(preprocess(sentence,no_stopwords=True,bigram=bigram,trigram=trigram))
        f.write('\n')


'''
# for phrase training only

with open('./data/text_cleaned.txt','w') as f:
    for sentence in sentences:
        f.write(preprocess(sentence,no_stopwords=True))
        f.write('\n')
'''