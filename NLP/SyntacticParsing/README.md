MVA - Algorithms for Speech and NLP TD 4
========================================

Author: Michael Ramamonjisoa

Tested on python=3

The tgz archive contains the IPython notebook TD3.ipynb, which contains all the code, comments, and experiments.

The $CONTEXT2VECDIR variable must point to the root directory with following content:

The working directory must contain the following:
----README.md
----sequoia-corpus+fct.mrg_strict
----TP4.ipynb
----script.py
----parse.sh

The shell script parse.sh is executed as follow
$ sh parse.sh [no arguments]

The user is then asked to enter tokenize sentences in the standard input.
Those sentences must be tokenize as such:
- exactly ONE blank space between each word
- one sentence per line

Here are examples of sentences which should work:

C' est enfin terminé !
Le procès de première instance .
Vous n' êtes pas prêts .
La terre est ronde .
