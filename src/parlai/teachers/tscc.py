#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard
import collections
import csv
import itertools
import json
import re
import os

# third
from parlai.core.teachers import register_teacher, DialogTeacher


__author__ = "Anaïs Tack"
__credits__ = ["Anaïs Tack", "Chris Piech"]
__copyright__ = "Copyright 2022, Anaïs Tack"
__license__ = "CC BY NC-SA 4.0"
__version__ = "1.0.0"
__maintainer__ = "Anaïs Tack"
__email__ = "atack@cs.stanford.edu"


ROLE_TEACHER = "teacher"
ROLE_STUDENT = "student"

TASK = 'TSCC'


def _make_dialogic_pairs(role_first=None):
    cache = dict(counter=0)

    def func(role):
        if role == role_first:
            cache['counter'] += 1
            return cache['counter']
        else:
            return cache['counter']

    return func


def _concat_turns(turns, use_edits=False):
    turns = list(turns)
    if use_edits:
        turn_sq = '\n'.join(t.clean(t.edited) if t.edited else t.clean(
            t.anonymised) for t in turns)
    else:
        turn_sq = '\n'.join(t.clean(t.anonymised) for t in turns)
    return turn_sq


class Chat(object):

    def __init__(self, turns) -> None:
        super().__init__()
        self._turns = collections.OrderedDict({t.number: t for t in turns})

    @property
    def turns(self):
        return self._turns.values()

    def get_turn(self, number):
        return self._turns.get(number)

    @classmethod
    def from_tsv(cls, filename):
        with open(filename) as fh:
            reader = csv.reader(fh, delimiter='\t')
            next(reader, None)  # read header
            turns = [Turn.from_row(i, line) for i, line in enumerate(reader)]
        return cls(turns)


class Turn(object):

    NA_VAL = "NA"
    ROLE_TEACHER = ROLE_TEACHER
    ROLE_STUDENT = ROLE_STUDENT

    def __init__(self,
                 line_idx,
                 timestamp,
                 user_id,
                 role,
                 turn_number,
                 anonymised,
                 edited,
                 responding_to=None,
                 sequence=None,
                 seq_type=None,
                 focus=None,
                 resource=None,
                 assessment=None) -> None:
        super().__init__()
        self.line_idx = line_idx
        self.timestamp = timestamp
        self.user_id = user_id
        self.role = role
        self.number = int(turn_number)
        self.anonymised = anonymised
        self.edited = edited
        self.responding_to = responding_to
        self.sequence = sequence
        self.seq_type = seq_type
        self.focus = focus
        self.resource = resource
        self.assessment = assessment

    @property
    def text(self):
        return self.anonymised

    @classmethod
    def from_row(cls, line_idx, cols):
        return cls(line_idx,
                   *map(lambda x: None if x == cls.NA_VAL else x, cols))

    @staticmethod
    def clean(turn_str):
        turn_new = re.sub(r"<([A-Z]+)([ '][A-Z ']+)?>",
                          lambda m: m.group(1).lower(), turn_str)
        turn_new = turn_new
        return turn_new


@register_teacher(TASK)
class TSCCTeacher(DialogTeacher):

    def __init__(self, opt, shared=None):
        opt['datafile'] = list(
            map(lambda f: os.path.join(opt['datapath'], f),
                filter(lambda f: f.endswith('.tsv'),
                       os.listdir(opt['datapath']))))
        super().__init__(opt, shared)

    @property
    def has_history(self):
        return True

    def setup_data(self, datafiles):
        # TSCC has a list of datafiles instead of one datafile
        for datafile in datafiles:
            filename = os.path.basename(datafile)

            # Datafile tells us where to load from.
            print(f" ~~ Loading from {datafile} ~~ ")
            tscc_chat = Chat.from_tsv(datafile)

            # consider one turn as a change of role
            # group consecutive turns with the same role
            turns_by_role = itertools.groupby(
                tscc_chat.turns, key=lambda t: t.role)
            # make sure list of turns can be iterated several times
            turns_by_role = ((role, list(turns_grp))
                             for role, turns_grp in turns_by_role)

            # group turns by dialogic pairs with the same counter
            # the counter starts from 1
            # - if the chat does not start with role_first
            # - (e.g. student starts conversation)
            # the counter starts from 0
            # - if the chat does not start with role_first
            # - (e.g. teacher starts conversation)
            pair_func = _make_dialogic_pairs(role_first=ROLE_STUDENT)
            turns_by_pair = itertools.groupby(
                turns_by_role, key=lambda rt_pair: pair_func(rt_pair[0]))

            for i, (__, grouper) in enumerate(turns_by_pair):
                new = True if i == 0 else False  # is this a new episode?
                roles, turns = zip(*grouper)
                turns = tuple(turns)
                # base case: dialogic pair (student, teacher)
                if len(turns) == 2:
                    msg = dict(text=_concat_turns(turns[0]),
                               labels=[_concat_turns(turns[1])],)
                # special case:
                # the chat does not start with student or end with teacher
                # (a) the call is empty, but not the response
                # --> teacher is the first person who speaks
                # (b) the call is empty, but no response
                # --> student is the last person who speaks
                elif len(turns) == 1:
                    if roles[0] == ROLE_TEACHER:
                        msg = dict(labels=[_concat_turns(turns[0])])
                    else:
                        msg = dict(text=_concat_turns(turns[0]))
                else:
                    raise Exception(
                        "Number of turns in dialogic pair should be 2 (or 1), "
                        "not {}".format(len(turns)))

                # save reference to filename and lines
                msg['wherefrom'] = dict(
                    filename=filename,
                    line_idx=[turn.line_idx
                              for speaker in turns
                              for turn in speaker])

                # select only relevant messages
                if self.opt.get('selection'):
                    if filename in self.opt['selection'] and \
                            any(lidx in self.opt['selection'][filename]
                                for lidx in msg['wherefrom']['line_idx']):
                        msg['context'] = json.dumps(msg['wherefrom'])
                        yield msg, new
                # return all messages
                else:
                    yield msg, new
