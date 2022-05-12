The AI Teacher Test
===================

.. image:: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
   :target: https://creativecommons.org/licenses/by-nc-sa/4.0/
   :alt: License: CC BY-NC-SA 4.0

.. image:: https://img.shields.io/badge/version-1.0.0-blue
   :target: https://github.com/anaistack/ai-teacher-test/tree/main
   :alt: package version


This repository contains the code and data for the paper:

:Title:
   `The AI Teacher Test: Measuring the Pedagogical Ability of Blender and GPT-3 in Educational Dialogues <https://anaistack.github.io/papers/tack_ai_2022/>`_

:Authors:
   Anaïs Tack & Chris Piech

:Abstract:
   How can we test whether state-of-the-art generative models, such as Blender and GPT-3, are good AI teachers, capable of replying to a student in an educational dialogue? Designing an AI teacher test is challenging: although evaluation methods are much-needed, there is no off-the-shelf solution to measuring pedagogical ability. This paper reports on a first attempt at an AI teacher test. We built a solution around the insight that you can run conversational agents in parallel to human teachers in real-world dialogues, simulate how different agents would respond to a student, and compare these counterpart responses in terms of three abilities: speak like a teacher, understand a student, help a student. Our method builds on the reliability of comparative judgments in education and uses a probabilistic model and Bayesian sampling to infer estimates of pedagogical ability. We find that, even though conversational agents (Blender in particular) perform well on conversational uptake, they are quantifiably worse than real teachers on several pedagogical dimensions, especially with regard to helpfulness (Blender: ∆ ability = −0.75; GPT-3: ∆ ability = −0.93).

Dependencies
------------

Code
~~~~

The code in this repository depends on the `ParlAI <https://parl.ai>`_ framework, the `OpenAI API <https://openai.com/api/>`_, the `Hugging Face <https://huggingface.co>`_ transformers library, and the `Stan <https://mc-stan.org/users/interfaces/pystan.html>`_ library.

.. code:: bash

   pip install -r src/requirements.txt

Data
~~~~

The data in this repository depends on student-teacher utterances coming from two datasets.
Because of copyright reasons, these texts were removed from the repository and replaced by the tag ``{COPYRIGHTED-TEXT}``.
In order to repopulate the data, you must:

1. Download the `Teacher-Student Chatroom Corpus <https://aclanthology.org/2020.nlp4call-1.2.pdf>`_. 
   Put the ``*.tsv`` files into `data/0_datasets/tscc/ <data/0_datasets/tscc>`_.
2. Download the `Educational Uptake Dataset <https://github.com/ddemszky/conversational-uptake>`_. 
   Put ``uptake_data.csv`` into `data/0_datasets/uptake/ <data/0_datasets/uptake>`_.
3. Run the following commands to repopulate the data with missing utterances and prompts.

   .. code:: bash

      python -m src.utils.repopulate -t TSCC -d data/0_datasets/tscc
      python -m src.utils.repopulate -t EduUptake -d data/0_datasets/uptake

Method
------

Simulating Agent Responses
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Download the pre-trained models into ``downloads/models/``.

   .. code:: bash

      python -m src.parlai.scripts.download_models downloads/ blender/blender_90M blender/blender_400Mdistill blender/blender_3B blender/blender_9B 

2. Run a Blender model on the data. For example:

   .. code:: bash
      
      python -m src.parlai.scripts.run -t TSCC -d data/0_datasets/tscc/ -M downloads/models -m blender/blender_9B -O results/
      python -m src.parlai.scripts.run -t EduUptake -d data/0_datasets/uptake/ -M downloads/models -m blender/blender_9B -O results/

3. Run a GPT-3 model on the data. For example:

   .. code::  bash

      python -m src.parlai.scripts.run -m src.parlai.models.gpt3:GPT3Davinci -o src/parlai/opts/gpt3.json -t TSCC -d data/0_datasets/tscc/ -O results/
      python -m src.parlai.scripts.run -m src.parlai.models.gpt3:GPT3Davinci -o src/parlai/opts/gpt3.json -t EduUptake -d data/0_datasets/uptake/ -O results/


Measuring Pedagogical Ability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Detect outliers among human raters.

   .. code:: bash

      python -m src.stan.bradley_terry data/2_comparisons/items.jsonl --per-rater

2. Estimate pedagogical abilities after outlier removal.

   .. code:: bash

      python -m src.stan.bradley_terry data/2_comparisons/items.jsonl --outliers data/2_comparisons/outliers.yaml


Citation
--------

More information can be found in `this paper <https://anaistack.github.io/assets/pdf/tack_ai_2022.pdf>`_. 
When using the data or code in your research or publication, please cite this paper as well.

.. code:: bibtex

   @inproceedings{tack_ai_2022,
      title = {The {{AI Teacher Test}}: {{Measuring}} the {{Pedagogical Ability}} of {{Blender}} and {{GPT-3}} in {{Educational Dialogues}}},
      booktitle = {The 15th {{International Conference}} on {{Educational Data Mining}}},
      author = {Tack, Ana{\"i}s and Piech, Chris},
      year = {2022},
      pages = {accepted},
      copyright = {All rights reserved}
      }

Acknowledgments
---------------

This research was funded by a fellowship of the `BAEF (Belgian American Educational Foundation) <https://www.baef.be>`_ and by a grant from `Stanford HAI <https://hai.stanford.edu>`_.

Changelog
---------

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`__,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`__.

[1.0.0] - 2022-05-10
~~~~~~~~~~~~~~~~~~~~

Added
   - Publication of data and code for the EDM 2022 conference

.. |copy|   unicode:: U+000A9 .. COPYRIGHT SIGN
