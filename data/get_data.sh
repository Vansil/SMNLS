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
mv snli_1.0 snli
# solve issues as described here: https://github.com/facebookresearch/SentEval/issues/56

# ELMo
echo "Getting ELMo"
mkdir -p elmo
wget -O elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
wget -O elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json

# Word-in-Context
echo "Getting Word-in-Context"
wget -O wic.zip https://pilehvar.github.io/wic/package/WiC_dataset.zip
unzip wic.zip -d wic/
rm wic.zip

# SentEval
echo "Getting SentEval"
git clone https://github.com/facebookresearch/SentEval.git
cd SentEval/
source activate dl
python setup.py install
source deactivate
cd data/downstream/
./get_transfer_data.bash

