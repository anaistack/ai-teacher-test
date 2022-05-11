#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard
from os.path import abspath, dirname, join


__author__ = "Anaïs Tack"
__credits__ = ["Anaïs Tack", "Chris Piech"]
__copyright__ = "Copyright 2022, Anaïs Tack"
__license__ = "CC BY NC-SA 4.0"
__version__ = "0.0.0"
__maintainer__ = "Anaïs Tack"
__email__ = "atack@cs.stanford.edu"


DATA_DIR = abspath(join(dirname(__file__), '..', 'data'))

GENERATIONS_DIR = join(DATA_DIR, '1_generations')
COMPARISONS_DIR = join(DATA_DIR, '2_comparisons')
ABILITIES_DIR = join(DATA_DIR, '3_abilities')
