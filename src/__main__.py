import argparse
import logging
from .ag import apply_model, train_model, list_templates, show_template

if __name__ == "__main__":

    parser = argparse.ArgumentParser("pysegment")
    subparsers = parser.add_subparsers()

    list_parser = subparsers.add_parser("list")
    list_parser.set_defaults(func=list_templates)

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--template", dest="template", required=True, help="Grammar template name or file")
    show_parser.set_defaults(func=show_template)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--pycfg", dest="pycfg", default=".", help="Directory with py-cfg binaries")
    train_parser.add_argument("--num_iterations", dest="num_iterations", type=int, default=10)
    train_parser.add_argument("--limit", dest="limit", type=int, default=None, help="Limit the number of training lines")
    train_parser.add_argument("--anneal_iterations", dest="anneal_iterations", type=int, default=5)
    train_parser.add_argument("--anneal_initial", dest="anneal_initial", type=float, default=3.0)
    train_parser.add_argument("--anneal_final", dest="anneal_final", type=float, default=1.0)    
    train_parser.add_argument("--input", dest="input", required=True, help="Input text file for training")
    train_parser.add_argument("--template", dest="template", required=True, help="Grammar template name or file (see 'list' subcommand)")
    train_parser.add_argument("--lowercase", dest="lowercase", default=False, action="store_true", help="Whether to lower-case the data")
    train_parser.add_argument("--sentence", dest="sentence", default=False, action="store_true", help="Operate at sentence level rather than word level")
    train_parser.add_argument("--type_level", dest="type_level", default=False, action="store_true", help="Type-level")
    train_parser.add_argument("--output", dest="output", required=True, help="Output for trained model file")
    train_parser.set_defaults(func=train_model)
    
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--cky", dest="cky", default=".", help="Directory with cky binaries")
    apply_parser.add_argument("--model", dest="model", required=True, help="Trained model file")
    apply_parser.add_argument("--unseen_weight", dest="unseen_weight", default=0.1, help="Weight to assign characters not seen in training")
    apply_parser.add_argument("--input", dest="input", required=True, help="Input text file to be segmented")
    apply_parser.add_argument("--sentence", dest="sentence", default=False, action="store_true", help="Operate at sentence level rather than word level")    
    apply_parser.add_argument("--lowercase", dest="lowercase", default=False, action="store_true", help="Whether to lower-case the data")
    apply_parser.add_argument("--type_level", dest="type_level", default=False, action="store_true", help="Type-level")
    apply_parser.add_argument("--bpe", dest="bpe", default=False, action="store_true", help="Insert BPE-style '@@'-boundaries")
    apply_parser.add_argument("--output", dest="output", required=True, help="Segmented output file")    
    apply_parser.set_defaults(func=apply_model)
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    args.func(args)
