#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard
import argparse as ap

# third
from parlai.scripts.display_model import DisplayModel


__author__ = "Anaïs Tack"
__credits__ = ["Anaïs Tack", "Chris Piech"]
__copyright__ = "Copyright 2022, Anaïs Tack"
__license__ = "CC BY NC-SA 4.0"
__version__ = "1.0.0"
__maintainer__ = "Anaïs Tack"
__email__ = "atack@cs.stanford.edu"


def main(args):
    for model_name in args.models:
        DisplayModel.main(
            task=f"{args.task}",
            # this is a dummy datapath
            # to make sure the model is saved in the right directory
            # parlai saves a model in the task data directory
            datapath=f"{args.models_dir}",
            model_file=f"zoo:{model_name}/model")


def args_parser():
    parser = ap.ArgumentParser()
    parser.add_argument('models_dir', type=str,
                        help="absolute path to directory "
                             "where the models will be saved")
    parser.add_argument('models', type=str, nargs='+', metavar='model')
    parser.add_argument('-t', '--task', type=str, default='convai2')
    return parser


if __name__ == "__main__":
    parser = args_parser()
    args = parser.parse_args()
    main(args)
