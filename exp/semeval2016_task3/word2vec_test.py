from gensim.models.word2vec import Word2Vec


model = Word2Vec.load('./data/word2vec/w2v.model')
for i in range(50):
    print(model.index2word[i])


while True:
    text = raw_input('Enter: ')
    print(model.most_similar(positive=[text], negative=[], topn=5))
