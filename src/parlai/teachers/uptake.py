#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard
import collections
import csv
import os

# third
from parlai.core.teachers import register_teacher, DialogTeacher


__author__ = "Anaïs Tack"
__credits__ = ["Anaïs Tack", "Chris Piech"]
__copyright__ = "Copyright 2022, Anaïs Tack"
__license__ = "CC BY NC-SA 4.0"
__version__ = "0.0.0"
__maintainer__ = "Anaïs Tack"
__email__ = "atack@cs.stanford.edu"


TASK = "EduUptake"


class Chat(object):

    def __init__(self, pairs) -> None:
        super().__init__()
        self._pairs = collections.OrderedDict(
            {(p.obs_id, p.exchange_idx): p for p in pairs})

    @property
    def pairs(self):
        return self._pairs.values()

    def get_pair(self, obs_id, exchange_idx):
        return self._pairs.get(obs_id, exchange_idx)

    @classmethod
    def from_csv(cls, filename):
        with open(filename) as fh:
            reader = csv.reader(fh)
            next(reader, None)  # read header
            pairs = [Pair.from_row(i, line) for i, line in enumerate(reader)]
        return cls(pairs)


class Pair(object):

    NA_VAL = "NA"

    def __init__(self,
                 line_idx,
                 obs_id,
                 exchange_idx,
                 student_text,
                 teacher_text,
                 student_on_task,
                 student_on_task_num_agree,
                 student_on_task_majority,
                 student_on_task_zscore,
                 teacher_on_task,
                 teacher_on_task_num_agree,
                 teacher_on_task_majority,
                 teacher_on_task_zscore,
                 uptake,
                 uptake_num_agree,
                 uptake_majority,
                 uptake_zscore) -> None:
        super().__init__()
        self.line_idx = line_idx
        self.obs_id = obs_id
        self.exchange_idx = exchange_idx
        self.student_text = student_text
        self.teacher_text = teacher_text
        self.student_on_task = student_on_task
        self.student_on_task_num_agree = student_on_task_num_agree
        self.student_on_task_majority = student_on_task_majority
        self.student_on_task_zscore = student_on_task_zscore
        self.teacher_on_task = teacher_on_task
        self.teacher_on_task_num_agree = teacher_on_task_num_agree
        self.teacher_on_task_majority = teacher_on_task_majority
        self.teacher_on_task_zscore = teacher_on_task_zscore
        self.uptake = uptake
        self.uptake_num_agree = uptake_num_agree
        self.uptake_majority = uptake_majority
        self.uptake_zscore = uptake_zscore

    @classmethod
    def from_row(cls, line_idx, cols):
        return cls(
            line_idx, *map(lambda x: None if x == cls.NA_VAL else x, cols))


@register_teacher(TASK)
class UptakeTeacher(DialogTeacher):

    def __init__(self, opt, shared=None):
        opt['datafile'] = os.path.join(opt['datapath'], 'uptake_data.csv')
        super().__init__(opt, shared)

    @property
    def has_history(self):
        return False

    def setup_data(self, datafile):
        filename = os.path.basename(datafile)
        # print(f" ~~ Loading from {datafile} ~~ ")
        uptake_chat = Chat.from_csv(datafile)

        for pair in uptake_chat.pairs:
            new = True  # is this a new episode?
            msg = dict(text=pair.student_text, label=pair.teacher_text)

            # save reference to filename and lines
            msg['wherefrom'] = dict(
                filename=filename,
                line_idx=pair.line_idx)

            yield msg, new
