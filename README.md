# TDParse
[![Build Status](https://travis-ci.org/apmoore1/tdparse.png?branch=master)](https://travis-ci.org/apmoore1/tdparse)

Implementation of [TDParse](https://aclanthology.coli.uni-saarland.de/papers/E17-1046/e17-1046).

[Read the Docs styled documentation](https://apmoore1.github.io/tdparse/)

## [Running the notebooks](./notebooks)
We use [Python markdown](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/python-markdown/readme.html) in the notebooks which requires [jupyter](http://jupyter.org/) extension Python makdown. To install this run:
`bash jupyter_extensions.sh`
If an errors occur this could be due to jupyter directories being installed under root. This can be overcome by changing ownership of those directories to you user account or running the previous command as root. To find the directories jupyter uses run the following command:
`jupyter --paths`


### Testing
Using [pytest](https://docs.pytest.org/en/latest/contents.html) and linking it to [Travis CI](https://travis-ci.org/). All tests are stored within the [tests directory](./tests).

To run the tests:
`python3 -m pytest`

### Data
We test this method on the following datasets:
1. [Li Dong et al.](https://aclanthology.coli.uni-saarland.de/papers/P14-2009/p14-2009) which can be found [here](http://goo.gl/5Enpu7). However we did not use there original data as it had been pre-processed by what we believe to be Stanford tools as there are tokens such as `-RRB-` which should be a `)` and also the data had been tokenised therefore the wording should be `Meeting...` it is actually `Meeting ...` and due to these pre-processing problems we went with the following [dataset](https://github.com/bluemonk482/tdparse/tree/master/data/lidong) from [Bo Wang et al.](https://aclanthology.coli.uni-saarland.de/papers/E17-1046/e17-1046) which is the same data just without any pre-processing.

### Requirements
1. Python Tested with 3.6 and known to not work with anything less than 3.5.
2. pip3 install -r requirements.txt
3. To use the [Tweebo Parser](./tools/TweeboParser) requires *gcc* and *cmake*. Tweebo will also automatically install when you first import the dependency_parsers module. Therefore to install it before run the script [./tools/TweeboParser/install.sh](./tools/TweeboParser/install.sh). This will take at least 15 minutes and depends on your Internet connection.
4. Docker - This is required to run Stanford CoreNLP. To install docker on Ubuntu run [docker_install](./docker_install.sh)
5. To get Stanford CoreNLP running use this docker image

## Word vectors
There are a number of word vectors stored in this repository they are listed below:
1. Vectors used in [Vo and Zhang](https://www.ijcai.org/Proceedings/15/Papers/194.pdf) which are [Sentiment Specific Word Embeddings (SSWE)](./data/word_vectors/vo_zhang) from [Tang et al.](https://aclanthology.coli.uni-saarland.de/papers/P14-1146/p14-1146) and [Word2Vec embeddings](./data/word_vectors/vo_zhang) that were trained on 5 million Tweets that are at least 7 tokens and one of the following tokens [":)",": )",":-)",":(",": (",":-("].
2. [Pre-trained Glove Twitter vectors](https://github.com/stanfordnlp/GloVe), which are licensed under the [PDDL](https://opendatacommons.org/licenses/pddl/).
