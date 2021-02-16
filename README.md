# documentclassification

## Download test data

```bash
mkdir tmp
cd tmp
wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
tar xvzf ldcc*tar.gz
find ./ -name LICENSE.txt -type f | xargs rm -rf
cd -
```

## Setup python

```bash
pipenv install
```

## Generate word2vec model

```bash
python learn_word2vec.py
ls model/
```

## Generate class model

```bash
echo WIP
```
