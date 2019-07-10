# Hierarchical Multi-Task Learning for Modeling Meaning Variation in Context

This is our research project for the Statistical Methods for Natural Language Semantics course at the University of Amsterdam (UvA).

## Usage

```bash
bash ./data/get_data.sh
# manually obtain the WSJ Penn Treebank dataset and put it at `./data/penn/`
# penn.zip md5: c7d01df318cf95bf3427d83a32872bb3
conda env create -n dl -f environment.yml
conda env update -n dl -f environment.yml
source activate dl
pip install -r requirements.txt
```
