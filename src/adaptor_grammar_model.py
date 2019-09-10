import argparse
import subprocess
import logging
import gzip
import tempfile
import os
import os.path
import shlex
import sys
import re
from nltk import Tree


space = "0020"


def to_hex(s):
    return ["%.4x" % (ord(c)) for c in s]


def from_hex(s):
    return [chr(int(c, base=16)) for c in s]


def make_pcfg(ag_gram, ensure_characters=set(), unseen_weight=0.1):
    new_lines = []
    ag_rule_counts = {}
    seen_characters = set()
    for line in ag_gram.strip().split("\n"):
        if "-->" in line:
            m = re.match(r"^.*Char --> (.*)$", line)
            if m:
                seen_characters.add(m.group(1))
            new_lines.append(line)
        elif re.match(r"^\s*$", line):
            continue
        else:
            m = re.match(r"^\((.*?)\#\d+(.*)$", line)
            lhs = m.group(1)
            rhs = tuple([[y for y in x if y != ""][0] for x in re.findall(r"\(Char (....)\)|(\^\^\^)|(\$\$\$)", m.group(2))])
            key = (lhs, rhs)
            ag_rule_counts[key] = ag_rule_counts.get(key, 0) + 1
    for (lhs, rhs), count in ag_rule_counts.items():
        new_lines.append("{}\t{} --> {}".format(count, lhs, " ".join(rhs)))
    for char in (ensure_characters - seen_characters):
        new_lines.append("{}\tChar --> {}".format(unseen_weight, char))
    return "\n".join(new_lines)


def train(args):
    logging.info("Reading grammar template from %s", args.template)
    with gzip.open(args.template, "rt") if args.template.endswith("gz") else open(args.template, "rt") as ifd:
        template = ifd.read().strip()        
    words = {}
    unique_characters = set(["^^^", "$$$"])
    logging.info("Reading data from %s", args.input)
    with gzip.open(args.input, "rt") if args.input.endswith("gz") else open(args.input, "rt") as ifd:
        for i, line in enumerate(ifd):
            if args.limit == i:
                break
            line = (line.lower() if args.lowercase else line).strip()
            for word in ([line] if args.sentence else line.split()): #(line.lower() if args.lowercase else line).strip().split():
                word_characters = []
                for c in word:                    
                    c = "%.4x" % (ord(c))
                    unique_characters.add(c)
                    word_characters.append(c)
                word = " ".join(word_characters)
                words[word] = words.get(word, 0) + 1
    rules = []
    for c in unique_characters:
        rules.append("0 1 Char --> %s" % (c))
    grammar = "{}\n{}".format(template, "\n".join(rules))
    logging.debug("Grammar: %s", grammar)    
    try:
        _, igrammar_fname = tempfile.mkstemp()
        _, parses_fname = tempfile.mkstemp()
        _, trace_fname = tempfile.mkstemp()
        _, ogrammar_fname = tempfile.mkstemp()        
        logging.info("Writing grammar to temporary file %s", igrammar_fname)
        with open(igrammar_fname, "wt") as ofd:
            ofd.write(grammar)
        cmd = "{} {} -w 0.1 -d 100 -E -n {} -e 1 -f 1 -g 10 -h 0.1 -T {} -t {} -m {} -A {} -G {} -F {}".format(os.path.join(args.pycfg, "py-cfg-quad"),
                                                                                                               igrammar_fname,
                                                                                                               args.num_iterations,
                                                                                                               args.anneal_initial,
                                                                                                               args.anneal_final,
                                                                                                               args.anneal_iterations,
                                                                                                               parses_fname,
                                                                                                               ogrammar_fname,
                                                                                                               trace_fname)
        words = sorted(words.items(), key=lambda x : x[1])
        data = "\n".join(["^^^ {} $$$".format(w) for w, _ in words])
        logging.info("Running %s on %d sequences", cmd, len(words))        
        pid = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        pid.communicate(data.encode())
        with open(ogrammar_fname, "rt") as ifd:
            ogrammar = ifd.read() 
    finally:
        for fname in [igrammar_fname, parses_fname, trace_fname]:
            if os.path.exists(fname):
                os.remove(fname)
    with open(args.output, "wt") as ofd:
        ofd.write(ogrammar)


def tree_to_morphs(tree):
    if isinstance(tree, str):
        return []
    elif all([isinstance(c, str) for c in tree]):
        return ["".join([chr(int(c, base=16)) for c in tree.leaves()])]
    elif tree.label() == "Chars":
        return ["".join([chr(int(c, base=16)) for c in tree.leaves() if c not in ["^^^", "$$$"]])]
    else:
        return sum([tree_to_morphs(x) for x in tree], [])
    #return ["".join([chr(int(c, base=16)) for c in x.leaves() if c not in ["^^^", "$$$"]]) for x in tree if not isinstance(x, str)]


def apply(args):
    words = {}
    unique_characters = set(["^^^", "$$$"])
    with gzip.open(args.input, "rt") if args.input.endswith("gz") else open(args.input, "rt") as ifd:
        for line in ifd:
            line = (line.lower() if args.lowercase else line).strip()            
            for word in ([line] if args.sentence else line.split()): #(line.lower() if args.lowercase else line).strip().split():
                #for word in (line.lower() if args.lowercase else line).strip().split():
                word_characters = []
                for c in word:                    
                    c = "%.4x" % (ord(c))
                    word_characters.append(c)
                    unique_characters.add(c)
                word = " ".join(word_characters)
                words[word] = words.get(word, 0) + 1
    with gzip.open(args.grammar, "rt") if args.grammar.endswith("gz") else open(args.grammar, "rt") as ifd:
        grammar = make_pcfg(ifd.read(), unique_characters, args.unseen_weight)

    data = "\n".join(["^^^ {} $$$".format(w) for w in words.keys()])
    try:
        _, data_fname = tempfile.mkstemp()
        _, grammar_fname = tempfile.mkstemp()
        with open(grammar_fname, "wt") as ofd:
            ofd.write(grammar)
        with open(data_fname, "wt") as ofd:
            ofd.write(data)
        data = data.encode()
        cmd = "{} {} 0 {}".format(os.path.join(args.cky, "ncky"),
                                  data_fname,
                                  grammar_fname)
        pid = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        out, _ = pid.communicate(data)
    finally:
        for fname in [grammar_fname, data_fname]:
            if os.path.exists(fname):
                os.remove(fname)
    with gzip.open(args.output, "wt") if args.output.endswith("gz") else open(args.output, "wt") as ofd:
        for orig, line in zip(data.decode().split("\n"), out.decode().split("\n")):
            if not re.match(r"^\s*$", line):
                tr = Tree.fromstring(line)
                morphs = tree_to_morphs(tr)                
                ofd.write(" ".join([m for m in morphs if len(m) > 0]) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--pycfg", dest="pycfg", default=".", help="Directory with py-cfg binaries")
    train_parser.add_argument("--num_iterations", dest="num_iterations", type=int, default=10)
    train_parser.add_argument("--limit", dest="limit", type=int, default=None, help="Limit the number of training lines")
    train_parser.add_argument("--anneal_iterations", dest="anneal_iterations", type=int, default=5)
    train_parser.add_argument("--anneal_initial", dest="anneal_initial", type=float, default=3.0)
    train_parser.add_argument("--anneal_final", dest="anneal_final", type=float, default=1.0)    
    train_parser.add_argument("--input", dest="input", required=True, help="Input file")
    train_parser.add_argument("--template", dest="template", required=True, help="Grammar template file")
    train_parser.add_argument("--lowercase", dest="lowercase", default=False, action="store_true", help="Whether to lower-case the data")
    train_parser.add_argument("--sentence", dest="sentence", default=False, action="store_true", help="Operate at sentence level rather than word level")
    train_parser.add_argument("--type_level", dest="type_level", default=False, action="store_true", help="Type-level")
    train_parser.add_argument("--output", dest="output", required=True, help="Output file")
    train_parser.set_defaults(func=train)
    
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--cky", dest="cky", default=".", help="Directory with cky binaries")
    apply_parser.add_argument("--grammar", dest="grammar", required=True, help="Input file")
    apply_parser.add_argument("--unseen_weight", dest="unseen_weight", default=0.1, help="Weight to assign characters not seen in training")
    apply_parser.add_argument("--input", dest="input", required=True, help="Input file")
    apply_parser.add_argument("--sentence", dest="sentence", default=False, action="store_true", help="Operate at sentence level rather than word level")    
    apply_parser.add_argument("--lowercase", dest="lowercase", default=False, action="store_true", help="Whether to lower-case the data")
    apply_parser.add_argument("--type_level", dest="type_level", default=False, action="store_true", help="Type-level")
    apply_parser.add_argument("--bpe", dest="bpe", default=False, action="store_true", help="Insert BPE-style '@@'-boundaries")
    apply_parser.add_argument("--output", dest="output", required=True, help="Output file")    
    apply_parser.set_defaults(func=apply)
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    args.func(args)
