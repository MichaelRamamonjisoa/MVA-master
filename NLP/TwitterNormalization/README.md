MVA - Algorithms for Speech and NLP TD 3
========================================

## Assignment details
**The goal of this assignment is to develop a normalisation system that can change raw English tweets into (partially) normalised tweets, suitable for further NLP processing.**

Your system will have to use two types of information for performing this normalisation task:
- contextual information (the context surrounding a given position helps guessing which correct word are possible in this position)
- formal similarity iformation (a misspelled/non-standard word is usually similar to its correct counterpart)

For dealing with contextual information, you will use the [*context2vec* program](https://github.com/orenmel/context2vec).

For dealing with formal similarity, you will use the notion of edit distance as computed by the Levenshtein algorithm (or any improvement thereof that you would prefer). You are requested to code yourself the algorithm using dynamic programming (i.e. you are not allowed to use an existing package for computing edit distance).

Beyond these constraints, you are free to do whatever you feel useful:
- You can train context2vec models or use existing ones (if you want to re-train context2vec, you will need a training corpus, which you can create by using tweet corpus creation tools; alternatively, you can play with the corpus provided in this folder).
- You can develop a training corpus and learn whatever you want on this corpus (e.g. the way to combine contextual and formal similarity information), or you can use a fully unsupervised approach.
- You can also develop complementary resources: normalisation lexicon, word embeddings such as those provided by GloVe (tweet-based embeddings available, see [here](https://nlp.stanford.edu/projects/glove/)), etc.
- You can design ways to deal with one-to-many (e.g. _ttyl_ -> _talk to you later_)  and even many-to-many mappings, or chose to only deal with one-to-one mappings.

## Dependencies

- [*context2vec* program](https://github.com/orenmel/context2vec)
- Chainer v1.7
- NLTK

## Data

Use any text file containing tweets. The program will read tweets line by line.

The program expects the [*UkWac*](http://irsrv2.cs.biu.ac.il/downloads/context2vec/context2vec.ukwac.model.package.tar.gz) model in the /model folder. 


## How to run

Run the following command: 

- $ <b>./run_system.sh numTweets</b> 

where numTweets is the number of tweets to be normalized in the text corpus.

## My approach

Details of my approach to this problem can be found in the IPython notebook. 
However, here is a summary of my strategy to return clean tweets.

In brief, I am trying to normalize tweet so as to have no contractions, no abbreviations, no misspelled words. The tweet must look as much as possible as properly written English. All special characters (except hashtags) must be deleted, and I also remove URLs, non-ASCII text, and HTML.







