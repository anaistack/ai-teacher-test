#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard
import argparse as ap
import json
import os
from collections import defaultdict
from glob import glob
from os.path import join

# third
from parlai.core import teachers

# local
from ..parlai.teachers import tscc, uptake  # <-- this adds new parlai teachers
from ..constants import ABILITIES_DIR, GENERATIONS_DIR


__author__ = "Anaïs Tack"
__credits__ = ["Anaïs Tack", "Chris Piech"]
__copyright__ = "Copyright 2022, Anaïs Tack"
__license__ = "CC BY NC-SA 4.0"
__version__ = "1.0.0"
__maintainer__ = "Anaïs Tack"
__email__ = "atack@cs.stanford.edu"


def _get_wherefrom(dict_):
    filename = dict_['wherefrom']['filename']
    line_idx = dict_['wherefrom']['line_idx']
    line_idx = line_idx if not isinstance(line_idx, list) else tuple(line_idx)
    return filename, line_idx


def _add_prompt(openai_response):
    text = openai_response['choices'][0]['text']
    return text


def main(args):

    opt = dict(
        task=args.task,
        datapath=args.datapath,
        datatype='valid')

    agent = teachers.create_task_agent_from_taskname(opt).pop()

    # read dataset into ParlAI messages
    dataset = defaultdict(dict)
    for __ in range(agent.num_examples()):
        msg = agent.act()
        filename, line_idx = _get_wherefrom(msg)
        dataset[filename][line_idx] = msg

    # iterate over files with generations
    for world_log in map(lambda f: join(GENERATIONS_DIR, f),
                         filter(lambda f: f.endswith('.jsonl'),
                                os.listdir(GENERATIONS_DIR))):
        # read world log and insert dataset
        # create backup
        with open(world_log) as fh, open(world_log + '.bak', 'w') as bak:
            lines = []
            for line in fh:
                # write to backup
                bak.write(line)
                line = json.loads(line.strip())
                # add missing text
                for i in range(len(line['dialog'])):
                    # find where in the dataset this utterance is from
                    wherefrom = _get_wherefrom(line['dialog'][i][0])
                    filename, line_idx = wherefrom
                    # only consider dialogs that come from the current dataset
                    if filename in dataset and line_idx in dataset[filename]:
                        item = dataset[filename][line_idx]
                        # add student utterance (if present)
                        line['dialog'][i][0]['text'] = item.get('text')
                        # add teacher utterance (if present)
                        teacher = item.get('eval_labels', [])
                        line['dialog'][i][0]['eval_labels'] = teacher
                        # add prompt for GPT-3
                        openai_response = line['dialog'][i][1].get(
                            'openai_response')
                        if openai_response:
                            text = _add_prompt(openai_response)
                            openai_response['choices'][0]['text'] = text
                lines.append(line)

        # write lines to file
        if len(lines):
            with open(world_log, 'w') as fh:
                for line in lines:
                    fh.write(json.dumps(line) + '\n')

    # iterate over files with abilities
    for filename in glob(f"{ABILITIES_DIR}/*.jsonl"):
        # read world log and insert dataset
        # create backup
        with open(filename) as fh, open(filename + '.bak', 'w') as bak:
            lines = []
            for line in fh:
                # write to backup
                bak.write(line)
                line = json.loads(line.strip())
                if line['id'] == 'Teacher':
                    filename, line_idx = _get_wherefrom(line)
                    if filename in dataset and line_idx in dataset[filename]:
                        line['text'] = dataset[filename][line_idx]['eval_labels'][0]
                lines.append(line)

        # write lines to file
        if len(lines):
            with open(filename, 'w') as fh:
                for line in lines:
                    fh.write(json.dumps(line) + '\n')


def args_parser():
    parser = ap.ArgumentParser()
    parser.add_argument('-t', '--task', required=True)
    parser.add_argument('-d', '--datapath', required=True)
    return parser


if __name__ == "__main__":
    parser = args_parser()
    args = parser.parse_args()
    main(args)
