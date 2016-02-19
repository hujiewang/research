from gensim.models.phrases import Phrases
from gensim.models.word2vec import LineSentence

sentence_stream=LineSentence('./data/text_cleaned.txt')
bigram = Phrases(sentence_stream,threshold=50.0)
bigram.save('./data/bigram.dat')
trigram = Phrases(bigram[sentence_stream],threshold=50.0)
trigram.save('./data/trigram.dat')

