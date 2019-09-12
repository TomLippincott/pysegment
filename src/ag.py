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
from pkg_resources import resource_string, resource_listdir, resource_isdir


def list_templates(args):    
    templates = [os.path.splitext(os.path.basename(x))[0] for x in resource_listdir(__name__, "grammar_templates")]
    print("Included grammar templates: {}".format(", ".join(templates)))


def get_template(template):
    if os.path.exists(template):
        with gzip.open(template, "rt") if template.endswith("gz") else open(template, "rt") as ifd:
            retval = ifd.read().strip()
    else:
        templates = {os.path.splitext(os.path.basename(x))[0] : x for x in resource_listdir(__name__, "grammar_templates")}
        if template in templates:
            retval = resource_string(__name__, os.path.join("grammar_templates", templates[template])).decode("utf-8").strip()
        else:
            retval = None
    return retval            


def show_template(args):
    template = get_template(args.template)
    if template == None:
        print("'{}' is neither a file nor a built-in template name (run the 'list' subcommand to check available templates')".format(args.template))
    else:
        print(template)


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


def train_model(args):

    if not os.path.exists(args.pycfg):
        raise Exception("""
Could not find py-cfg-quad binary under '{}'!
Did you forget to install it, or specify the wrong path to the --pycfg switch?
See the pyseg README.md file for instructions.
""")

    logging.info("Reading grammar template from %s", args.template)
    template = get_template(args.template)
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
        return ["".join([chr(int(c, base=16)) for c in tree.leaves() if c not in ["^^^", "$$$"]])]
    elif tree.label() == "Chars":
        return ["".join([chr(int(c, base=16)) for c in tree.leaves() if c not in ["^^^", "$$$"]])]
    else:
        return sum([tree_to_morphs(x) for x in tree], [])
    #return ["".join([chr(int(c, base=16)) for c in x.leaves() if c not in ["^^^", "$$$"]]) for x in tree if not isinstance(x, str)]

def segment(segmentations, token, args):
    toks = segmentations.get(token, list(token))
    toks = ["{}@@".format(t) for t in toks[0:-1]] + [toks[-1]]
    return toks

def apply_model(args):

    if not os.path.exists(args.cky):
        raise Exception("""
Could not find ncky binary under '{}'!
Did you forget to install it, or specify the wrong path to the --cky switch?
See the pyseg README.md file for instructions.
""")

    words = {}
    word_lookup = {}
    unique_characters = set(["^^^", "$$$"])
    with gzip.open(args.input, "rt") if args.input.endswith("gz") else open(args.input, "rt") as ifd:
        for line in ifd:
            line = (line.lower() if args.lowercase else line).strip()            
            for word in ([line] if args.sentence else line.split()): #(line.lower() if args.lowercase else line).strip().split():
                word_characters = []
                for c in word:                    
                    c = "%.4x" % (ord(c))
                    word_characters.append(c)
                    unique_characters.add(c)
                cpword = "^^^ {} $$$".format(" ".join(word_characters))
                words[cpword] = words.get(cpword, 0) + 1
                word_lookup[cpword] = word
    with gzip.open(args.model, "rt") if args.model.endswith("gz") else open(args.model, "rt") as ifd:
        grammar = make_pcfg(ifd.read(), unique_characters, args.unseen_weight)


    data = "\n".join(words.keys()) #["^^^ {} $$$".format(w) for w in words.keys()])
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

    segmentations = {}
    for orig, line in zip(data.decode().split("\n"), out.decode().split("\n")):
        if not re.match(r"^\s*$", line):
            tr = Tree.fromstring(line)
            morphs = tree_to_morphs(tr)                
            segmentations[word_lookup[orig]] = morphs

    
    with gzip.open(args.output, "wt") if args.output.endswith("gz") else open(args.output, "wt") as ofd:
        with gzip.open(args.input, "rt") if args.input.endswith("gz") else open(args.input, "rt") as ifd:
            for line in ifd:
                segmented = " ".join(sum([segment(segmentations, t, args) for t in line.strip().split()], []))
                ofd.write(segmented + "\n")
