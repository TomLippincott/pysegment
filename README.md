# pysegment

This code wraps the implementation of adaptor grammars by Mark Johnson as an approach to unsupervised morphological analysis.

## Installation

Starting from an empty working directory, something like the following installs all that you need beyond standard tools:

```
wget http://web.science.mq.edu.au/~mjohnson/code/py-cfg-2013-09-23.tgz
wget http://web.science.mq.edu.au/~mjohnson/code/cky.tbz
tar xpfz py-cfg-2013-09-23.tgz
tar cpfj cky.tbz
cd py-cfg
make
cd ../cky
make
cd ..
pip install pyseg --user
```

You can then train a model by running:

```
pysegment train --input TRAINING_DATA --template simple_prefix_suffix --pycfg py-cfg/ --output model.txt --num_iterations 200 --anneal_iterations 50 --limit 10000
```

This only uses the most-frequent 10000 words from the training data, and saves the resulting model (a PCFG augmented with hyper-parameters and cache usage).  You can then apply it to test data by running:

```
pysegment apply --input TEST_DATA --model model.txt --cky cky/ --output segs.txt
```
