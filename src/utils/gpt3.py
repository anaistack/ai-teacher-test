#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard
import os
import re

# third
import openai
from transformers import GPT2TokenizerFast


__author__ = "Anaïs Tack"
__credits__ = ["Anaïs Tack", "Chris Piech"]
__copyright__ = "Copyright 2022, Anaïs Tack"
__license__ = "CC BY NC-SA 4.0"
__version__ = "0.0.0"
__maintainer__ = "Anaïs Tack"
__email__ = "atack@cs.stanford.edu"


openai.api_key = os.getenv("OPENAI_API_KEY")


# maximum context length for the models
# considering both the input (prompt) and output (completion) tokens
MAX_CONTEXT_LEN = 2048

# models
DAVINCI = 'davinci'
CURIE = 'curie'
BABBAGE = 'babbage'
ADA = 'ada'
ENGINES = [DAVINCI, CURIE, BABBAGE, ADA]

# price per token (price rates are given per 1K tokens)
PRICING = {
    ADA: 0.0008 / 1000,
    BABBAGE: 0.0012 / 1000,
    CURIE: 0.0060 / 1000,
    DAVINCI: 0.0600 / 1000,
}


tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')


def tokenize(prompt):
    return tokenizer.encode(prompt)


def request_completion(prompt, *args, **kwargs):
    response = openai.Completion.create(
        prompt=prompt,
        *args, **kwargs
    )
    return response


def count_prompt_tokens(prompt, stop):
    stopwords_count = 0
    tokens_count = 0

    # because the tokenizer does not handle long sequences (max. 1024 tokens)
    # split the text to different lines and feed them to the tokenizer
    # this also removes the newline as stopword without counting it
    for line in re.split(r'\n+', prompt):

        # remove stopwords and trailing space (chatbot completion prefixes)
        for stopword in stop or []:
            line, count = re.subn(r'{}((?<=\:) )?'.format(
                re.escape(stopword)), r'', line)
            stopwords_count += count

        # tokenize the clean string
        tokens_count += len(tokenize(line))

    return stopwords_count + tokens_count


def compute_price(engine, prompt, stop, n, max_tokens, *args, **kwargs):
    return PRICING[engine] * (
        count_prompt_tokens(prompt, stop) +
        n * max_tokens)
