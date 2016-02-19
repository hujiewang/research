from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re

cachedStopWords = stopwords.words("english")
tokenizer = RegexpTokenizer(r'\w+')

def preprocess(text, no_stopwords=False, bigram=None, trigram=None):
    text=text.rstrip('\n')
    text=text.rstrip('\t')
    text=text.rstrip('\n\t')
    text=text.lower()
    text=' '.join(tokenizer.tokenize(text))
    text=re.sub("[0-9]+",'NUMBER',text)
    if no_stopwords:
        text = ' '.join([word for word in text.split() if word not in stopwords.words("english")])
    if bigram is not None:
        text=' '.join(bigram[text.split()])
    if bigram is not None and trigram is not None:
        text=' '.join(trigram[bigram[text.split()]])
    return text
