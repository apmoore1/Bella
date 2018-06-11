# Bella
[![Build Status](https://travis-ci.org/apmoore1/Bella.svg?branch=master)](https://travis-ci.org/apmoore1/Bella)

Target Dependent Sentiment Analysis (TDSA) framework.


## Requirements and Installation
1. Python 3.6
2. `pip install bella-tdsa`
3. Install [docker](https://docs.docker.com/install/)
4. Start Stanford CoreNLP server: `docker run -p 9000:9000 -d --rm mooreap/corenlp`
5. Start the TweeboParser API server: `docker run -p 8000:8000 -d --rm mooreap/tweeboparserdocker`

To stop the docker servers running:

1. Find the name assigned to the docker image using: docker ps
2. Then stop the relevant docker image: docker stop name_of_image

**NOTE**
Both of these servers will run with as many threads as your machine has CPUs to limit this do the following:
1. For stanford: `docker run -p 9000:9000 -d --rm mooreap/corenlp -threads 6` will run it with 6 threads
2. For TweeboParser: `docker run -p 8000:8000 -d --rm mooreap/tweeboparserdocker --threads 6` will run it with 6 threads


## Dataset

All of the dataset are required to be downloaded and are not stored in this repository. We recomend using the [config file](./config.yaml) to state where the datasets are stored like we did but this is not a requirement as you can state where they are stored explictly in the code. For more details on the datasets and downloading them see the [dataset notebook](https://github.com/apmoore1/Bella/blob/master/notebooks/datasets.ipynb) The datasets used:
1. [SemEval 2014 Resturant dataset](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools). We used Train dataset version 2 and the test dataset.
2. [SemEval 2014 Laptop dataset](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools). We used Train dataset version2 and the test dataset.
3. [Election dataset](https://figshare.com/articles/EACL_2017_-_Multi-target_UK_election_Twitter_sentiment_corpus/4479563/1)
4. [Dong et al.](https://aclanthology.coli.uni-saarland.de/papers/P14-2009/p14-2009) [Twitter dataset](https://github.com/bluemonk482/tdparse/tree/master/data/lidong)
5. [Youtubean dataset](https://github.com/epochx/opinatt/blob/master/samsung_galaxy_s5.xml) [by Marrese-Taylor et al.](https://www.aclweb.org/anthology/W17-5213)
6. [Mitchell dataset](http://www.m-mitchell.com/code/MitchellEtAl-13-OpenSentiment.tgz) which was released with this [paper](https://www.aclweb.org/anthology/D13-1171).

**NOTE** Before using Mitchell and YouTuBean datasets please go through these pre-processing notebooks: [Mitchell](https://github.com/apmoore1/Bella/blob/master/notebooks/Mitchel%20et%20al%20dataset%20splitting.ipynb) [YouTuBean](https://github.com/apmoore1/Bella/blob/master/notebooks/YouTuBean%20dataset%20splitting.ipynb) for splitting their data and also in Mitchell case which train test split to use.

## Lexicons

These lexicons are required to be downloaded if you use any methods that require them. Please see the use of the [config file](./config.yaml) for stroing the location of the lexicons:
1. MPQA can be found [here](http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/)
2. NRC [here](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
3. Hu and Liu [here](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon)

## Word Vectors

All the word vectors are automatically downloaded for you and they are stored in the root directory called '.Bella' which is created in your user directory e.g. on Linux that would be `~/.Bella/`. The word vectors included in this repository are the following:
1. SSWE
2. Word Vectors trained on sentences that contain emojis
3. Glove Common Crawl
4. Glove Twitter
5. Glove Wiki Giga


## Model Zoo

The model zoo can be found in the ["model zoo"](https://github.com/apmoore1/Bella/tree/master/model%20zoo) folder.

## The notebooks

Can be found [here](./notebooks)

The best order to look at the notebooks is first look at the data with this [notebook](./notebooks/datasets.ipynb). Then looking at the [notebook](./notebooks/Model%20Example.ipynb) that describes how to load and use the saved models from the [model zoo](#model-zoo). Then go and explore the rest if you would like: 

1. The Mass evaluation notebooks are the following
   * [Mass Evaluation - TDParse](./notebooks/Mass%20Evaluation%20-%20TDParse.ipynb) for the Target dependent models
   * [Mass Evaluation - Target Dependent](./notebooks/Mass%20Evaluation%20-%20Target%20Dependent.ipynb) for the TDParse models
   * [Mass Evaluation LSTM](./notebooks/Mass%20Evaluation%20LSTM.ipynb) for the LSTM models
  All of these do not contain any analysis just demostartes how we gathered the results. Lastly they also create the model zoo. For the analysis of the Mass evaluations see [Mass Evaluation Result Analysis](./notebooks/Mass%20Evaluation%20Result%20Analysis.ipynb) notebook
2. For the analysis of the reproduction of the Target Dependent model of [Vo and Zhang](https://www.ijcai.org/Proceedings/15/Papers/194.pdf) see this [notebook](./notebooks/target_model.ipynb)
3. For the analysis of the reproduction of the TDParse model of [Wang et al.](https://aclanthology.coli.uni-saarland.de/papers/E17-1046/e17-1046) see this [notebook](./notebooks/TDParse.ipynb)
4. For the analysis of the reproduction of the LSTM models of [Tang et al.](https://www.aclweb.org/anthology/C16-1311) see this [notebook](./notebooks/LSTM.ipynb)
5. For the statistics of the datasets and where to find them see this [notebook](./notebooks/datasets.ipynb)
6. For the code on creating training and test splits for the YouTuBean dataset see this [notebook](./notebooks/YouTuBean%20dataset%20splitting.ipynb)
7. For the code on creating training and test splits for [Mitchell et al.](https://www.aclweb.org/anthology/D13-1171) dataset see this [notebook](./notebooks/Mitchel%20et%20al%20dataset%20splitting.ipynb)


