#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard
import argparse as ap
import json
import re
from datetime import datetime

# local
from ..teachers import tscc, uptake  # <-- this adds new parlai teachers
from ..scripts.eval_model import EvalModel


__author__ = "Anaïs Tack"
__credits__ = ["Anaïs Tack", "Chris Piech"]
__copyright__ = "Copyright 2022, Anaïs Tack"
__license__ = "CC BY NC-SA 4.0"
__version__ = "0.0.0"
__maintainer__ = "Anaïs Tack"
__email__ = "atack@cs.stanford.edu"


def main(args):

    model_short = re.split(r'[/:]', args.model_name)[-1]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{args.task}_{model_short}"

    kwargs = {}

    # allow extra initialization options
    if args.init_opt:
        kwargs['init_opt'] = args.init_opt
        kwargs['allow_missing_init_opts'] = True

    # model is stored locally
    if args.models_dir:
        model_file = f"{args.models_dir}/{args.model_name}/model"
        kwargs['model_file'] = model_file
    else:
        kwargs['model'] = args.model_name

    # generate results (except for dry run)
    if not args.dry_run:
        report_filename = f"{args.output_dir}/reports/{filename}.json"
        kwargs['report_filename'] = report_filename
        world_logs = f"{args.output_dir}/world_logs/{filename}.jsonl"
        kwargs['world_logs'] = world_logs

    out = EvalModel.main(
        task=args.task,
        datapath=args.datapath,
        dry_run=args.dry_run,
        selection=json.dumps(args.selection),
        resume_after=json.dumps(args.resume_after),
        **kwargs)

    # do not generate results on dry run
    if not args.dry_run:
        with open(f'{args.output_dir}/eval_out/{filename}.json', 'w') as fh:
            json.dump(out, fh)


def args_parser():
    parser = ap.ArgumentParser()
    parser.add_argument('-t', '--task', required=True)
    parser.add_argument('-d', '--datapath', required=True)
    parser.add_argument('-m', '--model-name', required=True)
    parser.add_argument('-O', '--output-dir', required=True)
    parser.add_argument('-M', '--models-dir')
    parser.add_argument('-o', '--init-opt')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--selection', type=json.loads, default=r'{}')
    parser.add_argument('--resume-after', type=json.loads, default=r'{}')
    return parser


if __name__ == "__main__":
    parser = args_parser()
    args = parser.parse_args()
    main(args)
