# Bella
[![Build Status](https://travis-ci.org/apmoore1/Bella.svg?branch=master)](https://travis-ci.org/apmoore1/Bella)

Target Dependent Sentiment Analysis (TDSA) framework.


## Requirements and Installation
1. Python 3.6
2. `pip install bella-tdsa`
3. Install [docker](https://docs.docker.com/install/)
4. Start Stanford CoreNLP server: `docker run -p 9000:9000 -d --rm mooreap/corenlp`
5. Start the TweeboParser API server: `docker run -p 8000:8000 -d --rm mooreap/tweeboparserdocker`

If you want to use the moses tokeniser that has been taken from the [Moses project](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/python-tokenizer/moses.py) the following will need to be installed:
1. `python -m nltk.downloader perluniprops`
2. `python -m nltk.downloader nonbreaking_prefixes`

The docker Stanford and Tweebo server are only required if you are going to use the TDParse methods/models or if you are going to use any of the Stanford Tools else you do not need them.

To stop the docker servers running:

1. Find the name assigned to the docker image using: docker ps
2. Then stop the relevant docker image: docker stop name_of_image

**NOTE**
Both of these servers will run with as many threads as your machine has CPUs to limit this do the following:
1. For stanford: `docker run -p 9000:9000 -d --rm mooreap/corenlp -threads 6` will run it with 6 threads
2. For TweeboParser: `docker run -p 8000:8000 -d --rm mooreap/tweeboparserdocker --threads 6` will run it with 6 threads


## Dataset

All of the dataset are required to be downloaded and are not stored in this repository. We recomend using the [config file](./config.yaml) to state where the datasets are stored like we did but this is not a requirement as you can state where they are stored explictly in the code. For more details on the datasets and downloading them see the [dataset notebook](https://github.com/apmoore1/Bella/blob/master/notebooks/datasets.ipynb) The datasets used:
1. [SemEval 2014 Resturant dataset](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools). We used Train dataset version 2 and the test dataset of which the gold standatd test can be found [here](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-test-data-gold-annotations/b98d11cec18211e38229842b2b6a04d77591d40acd7542b7af823a54fb03a155/).
2. [SemEval 2014 Laptop dataset](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools). We used Train dataset version2 and the test dataset of which the gold standard test can be found [here](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-test-data-gold-annotations/b98d11cec18211e38229842b2b6a04d77591d40acd7542b7af823a54fb03a155/).
3. [Election dataset](https://figshare.com/articles/EACL_2017_-_Multi-target_UK_election_Twitter_sentiment_corpus/4479563/1)
4. [Dong et al.](https://aclanthology.coli.uni-saarland.de/papers/P14-2009/p14-2009) [Twitter dataset](https://github.com/bluemonk482/tdparse/tree/master/data/lidong)
5. [Youtubean dataset](https://github.com/epochx/opinatt/blob/master/samsung_galaxy_s5.xml) [by Marrese-Taylor et al.](https://www.aclweb.org/anthology/W17-5213)
6. [Mitchell dataset](http://www.m-mitchell.com/code/MitchellEtAl-13-OpenSentiment.tgz) which was released with this [paper](https://www.aclweb.org/anthology/D13-1171).

**NOTE** Before using Mitchell and YouTuBean datasets please go through these pre-processing notebooks: [Mitchell](https://github.com/apmoore1/Bella/blob/master/notebooks/Mitchel%20et%20al%20dataset%20splitting.ipynb) [YouTuBean](https://github.com/apmoore1/Bella/blob/master/notebooks/YouTuBean%20dataset%20splitting.ipynb) for splitting their data and also in Mitchell case which train test split to use.

## Lexicons

These lexicons are required to be downloaded if you use any methods that require them. Please see the use of the [config file](./config.yaml) for storing the location of the lexicons:
1. MPQA can be found [here](http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/)
2. NRC [here](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
3. Hu and Liu [here](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon)

## Word Vectors

All the word vectors are automatically downloaded for you and they are stored in the root directory called `.Bella/Vectors` which is created in your user directory e.g. on Linux that would be `~/.Bella/Vectors/`. The word vectors included in this repository are the following:
1. SSWE
2. Word Vectors trained on sentences that contain emojis
3. Glove Common Crawl
4. Glove Twitter
5. Glove Wiki Giga


## Model Zoo

The model zoo can be found on the Git Lab repository [here](https://delta.lancs.ac.uk/mooreap/bella-models).

These models can be automatically downloaded through the code like the word vectors and stored in the `.Bella/Models` directory which is automatically placed in your home directory for instance on Linux that would be `~/.Bella/Models`. An example of how to download and use a model is shown below:
```python
from bella import helper
from bella.models.target import TargetDep

target_dep = helper.download_model(TargetDep, 'SemEval 14 Restaurant')
test_example_multi = [{'text' : 'This bread is tasty but the sauce is too rich', 'target': 'sauce', 
                       'spans': [(28, 33)]}]

target_dep.predict(test_example_multi)
```
This example will download the Target Dependent model which is from [Vo and Zhang](https://www.ijcai.org/Proceedings/15/Papers/194.pdf) paper that has been trained on the SemEval 2014 Resturant data and predict the sentiment of **sauce** from that example. As you can see the example is not simple as it has two different sentiments within the same sentence with two targets; 1. **bread** with a positive sentiment and 2. **sauce** which has a negative sentiment of which that target is the one being predicted for in this example.

To see a more in depth guide to the pre-trained models and output from them go to [this notebook](./notebooks/Pre-Trained%20Model%20Example.ipynb).

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
8. Pre-Trained Model examples [notebook](./notebooks/Pre-Trained%20Model%20Example.ipynb)

## Docker Servers

Both Tweebo and Stanford by default will run on your localhost and port 8000 and 9000 respectively by default. If you would like to run them on a different *port* or *hostname* you can change the `.Bella/config.yaml` file which is created in your local home directory the first time you run something **successfully** through the Stanford or Tweebo tools. The file once created which is only done automatically for you when you **successfully** run something through the Stanford or Tweebo tools will look like this which is a yaml formatted file:
```yaml
tweebo_parser:
  hostname: 0.0.0.0
  port: 7000
stanford_core_nlp:
  hostname: http://localhost
  port: 8000
```
If you want Tweebo or Stanford to run on a different *port* or *hostname* just change this file. For instance the example shown above is different to default as Stanford is running on *port* 8000 and not 9000 and Tweebo is running on *port* 7000 instead of 8000.

If you would like the tools to run on different *hostname* and *port* from the start without having to **successfully** run them through the tools before hand just create this file `.Bella/config.yaml` in your local home directory with the same structure as the example but with the *hostname* and *port* you want to use.

## Different Licenses

As we use a lot of tools we list here if any of the tools that we use are licensed under a different license to that of this repository:
1. The Moses tokeniser is licensed under [GNU Lesser General Public License version 2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html) or, at your option, any later version.
