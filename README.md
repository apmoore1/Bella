# TDParse
[![Build Status](https://travis-ci.org/apmoore1/tdparse.png?branch=master)](https://travis-ci.org/apmoore1/tdparse)

Implementation of [TDParse](https://aclanthology.coli.uni-saarland.de/papers/E17-1046/e17-1046).

[Read the Docs styled documentation](https://apmoore1.github.io/tdparse/)


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
3. To use the [Tweebo Parser](./tools/TweeboParser) requires *gcc* and *cmake*
