# pyseg

This code wraps the implementation of adaptor grammars by Mark Johnson as an approach to unsupervised morphological analysis.

## Installation

Starting from an empty working directory, something like the following installs all that you need beyond standard tools:

```
pip install pipenv --user
wget http://web.science.mq.edu.au/~mjohnson/code/py-cfg-2013-09-23.tgz
wget http://web.science.mq.edu.au/~mjohnson/code/cky.tbz
git clone ssh://git@gitlab.hltcoe.jhu.edu:12321/lippincott/pyseg.git
tar xpfz py-cfg-2013-09-23.tgz
tar cpfj cky.tbz
cd py-cfg
make
cd ../cky
make
cd ../pyseg
pipenv install --skip-lock
```

You can then train a model by running:

```
pipenv run -- python src/adaptor_grammar_model.py train --input TRAINING_DATA --template grammar_templates/simple_prefix_suffix.txt --pycfg ../py-cfg/ --output model.txt --num_iterations 200 --anneal_iterations 50 --limit 10000
```

This only uses the most-frequent 10000 words from the training data, and saves the resulting model (a PCFG augmented with hyper-parameters and cache usage).  You can then apply it to test data by running:

```
pipenv run -- python src/adaptor_grammar_model.py apply --input TEST_DATA --grammar grammar_en.txt  --cky ../cky/ --output segs.txt
```
