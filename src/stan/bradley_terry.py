#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard
import argparse as ap
import collections
import json
import random
from os.path import dirname, join

# third
import arviz
import numpy as np
import pandas as pd
import stan
import yaml
from tqdm import tqdm


__author__ = "Anaïs Tack"
__credits__ = ["Anaïs Tack", "Chris Piech"]
__copyright__ = "Copyright 2022, Anaïs Tack"
__license__ = "CC BY NC-SA 4.0"
__version__ = "0.0.0"
__maintainer__ = "Anaïs Tack"
__email__ = "atack@cs.stanford.edu"


with open(join(dirname(__file__), 'bradley-terry-bayesian.stan')) as fh:
    bradley_terry_bayesian = fh.read()


def bradley_terry(df):
    data = {
        'K': len(np.unique(df[['player_i', 'player_j']].values)),
        'N': len(df),
        'i': list(df['player_i'].values),
        'j': list(df['player_j'].values),
        'y': list(df['i_gt_j'].values),
    }

    params_0 = ["alpha_0"]
    params = ["alpha", "ranking", "probability"]

    posterior = stan.build(bradley_terry_bayesian, data=data, random_seed=0)
    fit = posterior.sample()

    # compute summary statistics (mean, credible interval)
    summary = arviz.summary(fit, hdi_prob=.95, var_names=params_0 + params)

    # extract statistics for alphas and rankings
    # go back to 0-based index
    indices = sorted(map(lambda i: i-1, set(data['i'] + data['j'])))
    values = {n: {f'{param}_'+k: v
                  for param in params
                  for k, v in summary.loc[f'{param}[{n}]'].to_dict().items()}
              for n in indices}
    for param in params_0:
        for n in indices:
            for k, v in summary.loc[param].to_dict().items():
                values[n][f'{param}_'+k] = v

    return tuple(values.items())


def load_data(players, games):
    data = []
    for game in games:
        player_a, player_b, a_gt_b = game
        player_i = players.index(player_a) + 1
        player_j = players.index(player_b) + 1

        # transform i > j to integer score
        # if i > j is None (ties), set it to random uniform
        if a_gt_b is not None:
            i_gt_j = int(a_gt_b)
            data.append((player_i, player_j, i_gt_j))
        else:
            i_gt_j = int(random.choice([True, False]))
            data.append((player_i, player_j, i_gt_j))
    df = pd.DataFrame.from_records(
        data, columns=('player_i', 'player_j', 'i_gt_j'))
    return df


def iter(games_list):
    for games in games_list:
        data = bradley_terry.load_data(games)
        params = bradley_terry.bradley_terry(data)
        yield params


def compute_bradley_terry(players, games):
    data = load_data(players, games)
    params = bradley_terry(data)
    return params


def compute_per_item(task, players, items, outliers=None):
    outliers = outliers if outliers else {}

    def filter_out(dict_):
        return filter(lambda kv: kv[0] not in outliers,
                      dict_.items())

    # compute ability parameters
    ability = [dict(task=i['task'],
                    filename=i['filename'],
                    line_idx=str(i['line_idx']),
                    model=players[n], attribute=attr, **params)
               for i in filter(lambda i: task in i, items)
               for attr, responses in i[task]['attributes'].items()
               for n, params in compute_bradley_terry(
                   players,
                   map(lambda kv: kv[1],
                       filter_out(responses)))]

    return pd.DataFrame.from_records(ability)


def compute_per_rater(task, players, items):
    # reverse the usual computation
    # compute ability per rater
    raters = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(list))))
    for item in filter(lambda i: task in i, items):
        for attr, responses in item[task]['attributes'].items():
            for rater, response in responses.items():
                raters[rater][task]['attributes'][attr].append(response)

    ability = [dict(rater=rater, model=players[n], attribute=attr, **params)
               for rater in tqdm(sorted(raters)[60:], ascii=True)
               for attr, responses in raters[rater][task]['attributes'].items()
               for n, params in compute_bradley_terry(players, responses)]

    return pd.DataFrame.from_records(ability)


def main(args):
    task = 'comparisons'
    items = [json.loads(line)
             for line in map(str.strip,  args.responses_jsonl)]

    outliers = yaml.safe_load(args.outliers) if args.outliers else None

    # players are the list of models
    players = {i[task]['attributes'][attr][rater][n]
               for i in items
               if task in i
               for attr in i[task]['attributes']
               for rater in i[task]['attributes'][attr]
               for n in range(2)}
    players = list(sorted(players))

    if args.per_rater:
        df = compute_per_rater(task, players, items)
    else:
        df = compute_per_item(task, players, items, outliers=outliers)

    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('responses_jsonl', type=ap.FileType('r'))
    parser.add_argument('-o', '--output-file')
    parser.add_argument('--per-rater', action='store_true', default=False)
    parser.add_argument('--outliers', type=ap.FileType('r'), default=None)
    args = parser.parse_args()
    main(args)
