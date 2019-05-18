#!/bin/bash

# GloVe
if [ ! -d glove ]; then
    echo "Getting GloVe"
    wget -O glove.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip glove.zip -d glove/
    rm glove.zip
fi

# SNLI
if [ ! -d snli ]; then
    echo "Getting SNLI"
    wget -O snli.zip https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    unzip snli.zip
    rm snli.zip
    mv snli_1.0 snli
    # solve issues as described here: https://github.com/facebookresearch/SentEval/issues/56
fi

# ELMo
if [ ! -d elmo ]; then
    echo "Getting ELMo"
    mkdir -p elmo
    wget -O elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
    wget -O elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
fi

# Word sense disambiguation
if [ ! -d word-sense-disambiguation ]; then
    echo "Fetching word sense disambiguation dataset."
    git clone https://github.com/google-research-datasets/word_sense_disambigation_corpora.git word-sense-disambiguation
fi

# Word-in-Context
if [ ! -d wic ]; then
    echo "Getting Word-in-Context"
    wget -O wic.zip https://pilehvar.github.io/wic/package/WiC_dataset.zip
    unzip wic.zip -d wic/
    rm wic.zip
fi

# VUA sequential metaphor detection corpus.
if [ ! -d vua-sequence ]; then
    echo "Getting VUA metaphor detection corpus"
    mkdir -p vua-sequence
    wget -O vua-sequence/test.csv https://raw.githubusercontent.com/gao-g/metaphor-in-context/master/data/VUAsequence/VUA_seq_formatted_test.csv
    wget -O vua-sequence/train.csv https://raw.githubusercontent.com/gao-g/metaphor-in-context/master/data/VUAsequence/VUA_seq_formatted_train.csv
    wget -O vua-sequence/validation.csv https://raw.githubusercontent.com/gao-g/metaphor-in-context/master/data/VUAsequence/VUA_seq_formatted_val.csv
fi

# SentEval
if [ ! -d SentEval ]; then
    echo "Getting SentEval"
    git clone https://github.com/facebookresearch/SentEval.git
    cd SentEval/
    source activate dl
    python setup.py install
    source deactivate
    cd data/downstream/
    ./get_transfer_data.bash
fi

# Penn Treebank Wall Street Journal
if [ ! -d penn ]; then
	echo "Penn Treebank Wall Street Journal portion is not freely available. Please download manually and make sure the portion is available under path data/penn/wsj/"
fi
