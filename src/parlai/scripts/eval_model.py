#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard
import json

# third
from parlai.scripts.eval_model import EvalModel


__author__ = "Anaïs Tack"
__credits__ = ["Anaïs Tack", "Chris Piech"]
__copyright__ = "Copyright 2022, Anaïs Tack"
__license__ = "CC BY NC-SA 4.0"
__version__ = "1.0.0"
__maintainer__ = "Anaïs Tack"
__email__ = "atack@cs.stanford.edu"


class EvalModel(EvalModel):

    @classmethod
    def setup_args(cls):
        parser = super().setup_args()
        parser.add_argument(
            '--dry-run',
            action='store_true')
        parser.add_argument(
            '--selection',
            type=json.loads,
            default=r'{}')
        parser.add_argument(
            '--resume-after',
            type=json.loads,
            default=r'{}')

        return parser
