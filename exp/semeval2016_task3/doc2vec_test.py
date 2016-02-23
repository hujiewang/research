from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument
model=Doc2Vec.load('data/word2vec/d2v.model')
'''
for i in range(50):
    print(model.index2word[i])
'''
doc = TaggedLineDocument('./data/text_cleaned_phrase.txt')
t={}
for d in doc:
    t[d[1][0]]=d[0]
for a in model.docvecs.most_similar(0,topn=20):
    print(" ".join(t[a[0]]))