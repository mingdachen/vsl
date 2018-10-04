# VSL

A PyTorch implementation of "[Variational Sequential Labelers for Semi-Supervised Learning](http://ttic.uchicago.edu/~mchen/papers/mchen+etal.emnlp18.pdf)" (EMNLP 2018)


## Prerequisites

- Python 3.5
- PyTorch 0.3.0
- Scikit-Learn
- NumPy

## Data and Pretrained Embeddings

Download: [Twitter](https://code.google.com/archive/p/ark-tweet-nlp/downloads), [Universal Dependencies](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1827?show=full), [Embeddings (for Twitter and UD)](https://drive.google.com/drive/folders/1oie43_thsbhhoUsOHlkyKj2iMpFNOrgA?usp=sharing)

Run `process_{ner,twitter,ud}_data.py` first to generate `*.pkl` files and then use it as input for `vsl_{g,gg}.py`.

## Citation

```
@inproceedings{mchen-variational-18,
  author    = {Mingda Chen and Qingming Tang and Karen Livescu and Kevin Gimpel},
  title     = {Variational Sequential Labelers for Semi-Supervised Learning},
  booktitle = {Proc. of {EMNLP}},
  year      = {2018}
}
```
