#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
usage: python train.py wiki.en.text wiki.en.text.model wiki.en.text.vector
'''
import logging
import os.path
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    outp1='./data/word2vec/w2v.model'
    outp2='./data/word2vec/w2v.vec'
    sentences=LineSentence('./data/text_cleaned_phrase.txt')
    model = Word2Vec(size=30, window=5, min_count=5,max_vocab_size=50000,workers=multiprocessing.cpu_count())
    model.build_vocab(sentences)
    for i in range(30):
        model.train(sentences)
    # trim unneeded model memory = use(much) less RAM
    #model.init_sims(replace=True)
    model.save(outp1)
    model.save_word2vec_format(outp2, binary=False)