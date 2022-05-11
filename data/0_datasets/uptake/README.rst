Educational Uptake Dataset
==========================

- Download the data from `this repository <https://github.com/ddemszky/conversational-uptake>`_. 
- Put ``uptake_data.csv`` into this directory.

Citation
--------

.. code:: bibtex

    @inproceedings{demszky-etal-2021-measuring,
        title = "Measuring Conversational Uptake: A Case Study on Student-Teacher Interactions",
        author = "Demszky, Dorottya  and
            Liu, Jing  and
            Mancenido, Zid  and
            Cohen, Julie  and
            Hill, Heather  and
            Jurafsky, Dan  and
            Hashimoto, Tatsunori",
        booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
        month = aug,
        year = "2021",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.acl-long.130",
        doi = "10.18653/v1/2021.acl-long.130",
        pages = "1638--1653",
        abstract = "In conversation, uptake happens when a speaker builds on the contribution of their interlocutor by, for example, acknowledging, repeating or reformulating what they have said. In education, teachers{'} uptake of student contributions has been linked to higher student achievement. Yet measuring and improving teachers{'} uptake at scale is challenging, as existing methods require expensive annotation by experts. We propose a framework for computationally measuring uptake, by (1) releasing a dataset of student-teacher exchanges extracted from US math classroom transcripts annotated for uptake by experts; (2) formalizing uptake as pointwise Jensen-Shannon Divergence (pJSD), estimated via next utterance classification; (3) conducting a linguistically-motivated comparison of different unsupervised measures and (4) correlating these measures with educational outcomes. We find that although repetition captures a significant part of uptake, pJSD outperforms repetition-based baselines, as it is capable of identifying a wider range of uptake phenomena like question answering and reformulation. We apply our uptake measure to three different educational datasets with outcome indicators. Unlike baseline measures, pJSD correlates significantly with instruction quality in all three, providing evidence for its generalizability and for its potential to serve as an automated professional development tool for teachers.",
    }
