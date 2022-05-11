#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard
import json
import sys
import time

# third
import openai
from parlai.core.agents import Agent
from parse import parse

# local
from ...utils import gpt3


__author__ = "Anaïs Tack"
__credits__ = ["Anaïs Tack", "Chris Piech"]
__copyright__ = "Copyright 2022, Anaïs Tack"
__license__ = "CC BY NC-SA 4.0"
__version__ = "1.0.0"
__maintainer__ = "Anaïs Tack"
__email__ = "atack@cs.stanford.edu"


STUDENT_PREFIX = 'Student:'
TEACHER_PREFIX = 'Teacher:'
AI_PREFIX = 'AI:'

STOP = ["\n", f"{STUDENT_PREFIX}", f"{TEACHER_PREFIX}", f"{AI_PREFIX}"]

INSTRUCTIONS = "The following is a conversation with a teacher. "\
               "The teacher is polite, helpful, professional, "\
               "on topic, and factually correct."

TOKENS_ERROR = "This model's maximum context length is "\
               "{expected:d} tokens, however you requested "\
               "{observed:d} tokens ({prompt:d} in your prompt, "\
               "{completion:d} for the completion). "\
               "Please reduce your prompt or completion length."


class GPT3Agent(Agent):

    def __init__(self, opt):
        self.opt = opt

        self.config_gpt3 = dict(

            # https://beta.openai.com/docs/api-reference/completions/create#completions/create-engine_id
            engine=opt.get('gpt3_engine', gpt3.ADA),

            # https://beta.openai.com/docs/api-reference/completions/create#completions/create-max_tokens
            max_tokens=opt.get('gpt3_max_tokens', 16),

            # https://beta.openai.com/docs/api-reference/completions/create#completions/create-temperature
            temperature=opt.get('gpt3_temperature', 1),

            # https://beta.openai.com/docs/api-reference/completions/create#completions/create-top_p
            top_p=opt.get('gpt3_top_p', 1),

            # https://beta.openai.com/docs/api-reference/completions/create#completions/create-n
            n=opt.get('gpt3_n', 1),

            # https://beta.openai.com/docs/api-reference/completions/create#completions/create-stream
            stream=opt.get('gpt3_stream', False),

            # https://beta.openai.com/docs/api-reference/completions/create#completions/create-logprobs
            logprobs=opt.get('gpt3_logprobs', None),

            # https://beta.openai.com/docs/api-reference/completions/create#completions/create-echo
            echo=opt.get('gpt3_echo', False),

            # https://beta.openai.com/docs/api-reference/completions/create#completions/create-stop
            stop=opt.get('gpt3_stop', STOP),

            # https://beta.openai.com/docs/api-reference/completions/create#completions/create-presence_penalty
            presence_penalty=opt.get('gpt3_presence_penalty', 0),

            # https://beta.openai.com/docs/api-reference/completions/create#completions/create-frequency_penalty
            frequency_penalty=opt.get('gpt3_frequency_penalty', 0),

            # https://beta.openai.com/docs/api-reference/completions/create#completions/create-best_of
            best_of=opt.get('gpt3_best_of', 1),

            # https://beta.openai.com/docs/api-reference/completions/create#completions/create-logit_bias
            logit_bias=opt.get('gpt3_logit_bias', None),

        )

        # --- extra parameters for chatbot

        self.config_chat = dict(

            # the instructions that start the prompt
            instructions=opt.get('gpt3_instructions', INSTRUCTIONS),

            # the maximum length of the chatbot history
            # (in number of previous dialogic pairs)
            # the default (None) means the entire history will be considered
            max_history_len=opt.get('gpt3_max_history_len', None),

        )

        self.history = []

        self._exit = False
        self._resume_after = False

    @property
    def id(self):
        return 'GPT-3 {}'.format(self.config_gpt3['engine'].capitalize())

    @classmethod
    def restrict_history(cls,
                         context: str,
                         history: list,
                         turn: str,
                         max_history_len: int = None,
                         max_completion_len: int = 0,
                         penalty: int = 0,
                         *args, **kwargs):
        # if no maximum history is given, use the length of the history
        if max_history_len is None:
            max_history_len = len(history)
        # max_context_len is the total number of tokens accepted by the model
        # (including generated 'completion' tokens)
        max_prompt_len = gpt3.MAX_CONTEXT_LEN - max_completion_len
        # count number of tokens in chatbot context and most recent turn
        count = len(gpt3.tokenize(context)) + len(gpt3.tokenize(turn))
        # count Q & A prefixes (stopword)
        count += 2
        # iterate over previous history (from recent to old)
        for h, hist in enumerate(history):
            if h < max_history_len:
                q, a = hist.get('text', ''), hist.get('labels', [''])[
                    0] or hist.get('eval_labels', [''])[0]
                q_len = len(gpt3.tokenize(q)) if q else 0
                a_len = len(gpt3.tokenize(a)) if a else 0
                if (q or a) and count + q_len + a_len + 2 < max_prompt_len:
                    count += q_len + a_len
                    count += 2  # count Q & A prefixes (stopword)
                    yield q, a
                else:
                    break
            else:
                break

    @staticmethod
    def make_pair(turn,
                  answer,
                  prefix_t=STUDENT_PREFIX,
                  prefix_ai=TEACHER_PREFIX):
        pair = []
        if turn:
            pair.append(f"{prefix_t} {turn}")
        if answer:
            pair.append(f"{prefix_ai} {answer}")
        if not pair:
            raise Exception("Empty dialogic pair.")
        return '\n'.join(pair)

    @classmethod
    def make_prompt(cls,
                    observation,
                    history,
                    instructions=INSTRUCTIONS,
                    prefix_t=STUDENT_PREFIX,
                    prefix_ai=TEACHER_PREFIX,
                    *args, **kwargs):
        current = observation.get('text', '')

        # restrict most recent history to the maximum allowed length
        # counting all tokens including context and turn
        # [!] history must be a list starting from last turns to earliest turns
        history_res = list(
            cls.make_pair(turn, answer, prefix_t=prefix_t, prefix_ai=prefix_ai)
            for turn, answer in cls.restrict_history(instructions,
                                                     history,
                                                     current,
                                                     *args, **kwargs))

        # reverse history from first turns to last turns to build the prompt
        history = '\n'.join(reversed(history_res))

        # add preceding dialogue history
        if history:
            prompt = f"{instructions}\n\n{history}\n"\
                     f"{prefix_t} {current}\n{prefix_ai}"
        else:
            prompt = f"{instructions}\n\n{prefix_t} {current}\n{prefix_ai}"

        return prompt

    def act(self):

        skip = False

        # [!] skip to exit if there was a problem with GPT-3
        # without having to raise an exception or lose data
        if self._exit:
            skip = True

        # [!] skip this observation
        # if we want to focus on a selection only
        if self.opt['selection']:
            filename = self.observation['wherefrom']['filename']
            line_idx = self.observation['wherefrom']['line_idx']
            line_idx = line_idx if isinstance(line_idx, list) else [line_idx]
            if filename not in self.opt['selection']:
                skip = True
            elif all(i not in self.opt['selection'][filename]
                     for i in line_idx):
                skip = True

        # [!] skip this observation
        # if we need to resume from a previous item (resume run with errors)
        if self.opt['resume_after'] and self._resume_after is False:
            filename = self.observation['wherefrom']['filename']
            line_idx = self.observation['wherefrom']['line_idx']
            line_idx = line_idx if isinstance(line_idx, list) else [line_idx]
            # we haven't yet started to resume the run
            # now, we should resume after this observation
            if filename == self.opt['resume_after']['filename']\
                    and line_idx == self.opt['resume_after']['line_idx']:
                self._resume_after = True
            skip = True

        # prepare prompt
        instructions = self.config_chat['instructions']
        max_history_len = self.config_chat['max_history_len']
        max_completion_len = self.config_gpt3['max_tokens']

        prompt = self.make_prompt(
            self.observation,
            self.history,
            instructions=instructions,
            max_completion_len=max_completion_len,
            max_history_len=max_history_len)

        prompt_tokens = gpt3.count_prompt_tokens(
            prompt, self.config_gpt3['stop'])
        assert prompt_tokens + max_completion_len <= gpt3.MAX_CONTEXT_LEN

        # prepare request
        kwargs = dict(prompt=prompt, **self.config_gpt3)

        # try to make a request
        # gracefully manage errors such that no data will be lost
        done = False
        exit = False
        minute_elapsed = False
        retry_elapsed = False
        while not skip and not done and not exit:
            try:
                if self.opt['dry_run']:
                    sys.stdout.write(json.dumps(kwargs) + '\n')
                    response = dict(choices=[dict(text='')])
                else:
                    response = gpt3.request_completion(**kwargs)
            except openai.error.RateLimitError as e:
                if minute_elapsed:
                    sys.stderr.write(str(e) + "\n")
                    exit = True
                time.sleep(60)
                minute_elapsed = True
            except openai.error.InvalidRequestError as e:
                if retry_elapsed:
                    sys.stderr.write(str(e) + "\n")
                    exit = True
                params = parse(TOKENS_ERROR, str(e))
                # discount extra tokens from the maximum completion length
                extra_tokens = params['observed'] - params['expected']
                kwargs['max_tokens'] = kwargs['max_tokens'] - extra_tokens
                retry_elapsed = True
            else:
                done = True

        # [!] make sure to clear history correctly
        # [!] make sure history is added even when observation is skipped
        # append history
        if not self.observation['episode_done']:
            self.history.insert(0, self.observation)
        # reset history
        else:
            self.history = []

        # if there was a problem with GPT-3
        # continue exiting gracefully,
        # without raising an error that will lose data
        if exit:
            self._exit = True

        # if there was no problem with GPT-3
        # save the results
        if done:
            # extract the last text (= completion)
            # from the entire prompt echoed back
            full_text = response['choices'][0]['text']
            full_text = full_text.rsplit('\n', 1)[-1]
            parsed_text = parse(f'{TEACHER_PREFIX} ' + '{text}', full_text)
            text = parsed_text['text'] if parsed_text else ''

            sys.stderr.write(
                "[Done] " + json.dumps(self.observation['wherefrom']) + '\n')

            return dict(id=self.id, text=text, openai_response=response)
        # fallback: return an empty dictionary
        # to make sure there will be no error raised
        else:
            return {}


class GPT3Ada(GPT3Agent):

    def __init__(self, opt):
        super(GPT3Ada, self).__init__(opt)
        self.config_gpt3['engine'] = gpt3.ADA


class GPT3Babbage(GPT3Agent):

    def __init__(self, opt):
        super(GPT3Babbage, self).__init__(opt)
        self.config_gpt3['engine'] = gpt3.BABBAGE


class GPT3Curie(GPT3Agent):

    def __init__(self, opt):
        super(GPT3Curie, self).__init__(opt)
        self.config_gpt3['engine'] = gpt3.CURIE


class GPT3Davinci(GPT3Agent):

    def __init__(self, opt):
        super(GPT3Davinci, self).__init__(opt)
        self.config_gpt3['engine'] = gpt3.DAVINCI
