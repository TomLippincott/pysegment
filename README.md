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

## Usage

You can then train a model by running:

```
pysegment train --input TRAINING_DATA --template simple_prefix_suffix --pycfg py-cfg/ --output model.txt --num_iterations 200 --anneal_iterations 50 --limit 10000
```

This only uses the most-frequent 10000 words from the training data, and saves the resulting model (a PCFG augmented with hyper-parameters and cache usage).  You can then apply it to test data to get BPE-style output by running:

```
pysegment apply --input TEST_DATA --model model.txt --cky cky/ --output segmented.txt --bpe
```

There are several included adaptor grammar templates, you can list them with `pysegment list`, and print out the content of one of them with `pysegment show --template NAME`.  When training, you can either specify an included template, or a file of your own.

## Writing your own grammar template

Examining one of the included templates with `pysegment show --template simple_prefix_suffux` gives:

```
0 1 Word --> Prefix Stem Suffix
Prefix --> ^^^ Chars
Prefix --> ^^^
Stem --> Chars
Suffix --> Chars $$$
Suffix --> $$$
0 1 Chars --> Char Chars
0 1 Chars --> Char
```

This is exactly the format used by py-cfg, so see that documentation for in-depth details, but some broad points: rules starting with `0 1` are *not adapted*, while the rest are.  `pysegment` relies on three conventions, in addition to a well-formed grammar: 

  * `^^^` and `$$$` are special start and end symbols, respectively
  * All derivations lead to `Char` (or a specific character-sequence, see below)
  * Specific characters are indicated by their unicode hex value

The latter two points taken together mean, for example, one can hard-code language-specific information, like for English:

```
0 1 Suffix --> 0069 006e 0067
```

Where those hex values (note the convention is lowercase) correspond to "ing".  This, combined with manipulating the numeric values at the start of the rule, are ways to include extra knowledge from e.g. lexicons.  `pysegment` automatically creates productions from `Char` to each unique character seen in training, and when applied, adds additional low-probability productions for unseen characters to ensure complete coverage.
