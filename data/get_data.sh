#!/bin/bash

# GloVe
echo "Getting GloVe"
wget -O glove.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.zip -d glove/
rm glove.zip

# SNLI
echo "Getting SNLI"
wget -O snli.zip https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli.zip
rm snli.zip

# ELMo
echo "Getting ELMo"
mkdir -p elmo
wget -O elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
wget -O elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json




