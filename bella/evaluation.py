'''
Functions that allow evaluating model predictions easier, as well as custom
evaluation metrics specific for Target Based Sentiment Analysis e.g. the
:py:func:`bella.evaluation.distinct_sentiment_metrics` can return the
accuracy of the models predictions on samples that have a certain number of
distinct sentiments.

Functions:

1. score -- Given a metric, test data and predictions will evaluate the
   predictions based on the given metric.
2. evaluate_model -- Given a metric, dictionary of dataset names with test data
   and predictions will evaluate the predictions for each dataset.
3. evaluate_models -- Given a metric, dictionary of dataset names with test
   data and a dictionary of models and their predictions will evaluate the
   predictions for each dataset and for each model.
4. distinct_sentiment_metrics -- Custom metric function. It performs the
   `metric_func` on a subset of the test data where the samples have to come
   from sentences that contain a certain number of distinct_sentiments.
'''
from collections import defaultdict
import copy
from typing import Callable, Union, Dict

import numpy as np
import pandas as pd

import bella


def score(metric: Union[Callable[[np.ndarray, np.ndarray], float],
                        Callable[['bella.data_types.TargetCollection',
                                  np.ndarray], float]],
          test_data: 'bella.data_types.TargetCollection',
          predictions: np.ndarray,
          custom_metric: bool = False, **metric_kwargs) -> float:
    '''
    Given a metric, test data and predictions will evaluate the predictions
    based on the given metric.

    :param metric: Metric used to evaluate the predictions e.g.
                   :py:func:`sklearn.metrics.f1_score` or a custom function
                   like :py:func:`bella.evaluation.distinct_sentiment_metrics`
    :param test_data: The test data stored as a
                      :py:class:`bella.data_types.TargetCollection` object.
    :param predictions: Array of predictions for the test data
    :param custom_metric: If the metric being used has come from the Bella
                          package then it is custom and this should be True
                          else use the Default False.
    :param metric_kwargs: Keyword arguments to the metric function.
    :return: The value given back by the metric function.
    '''
    if custom_metric:
        return metric(test_data, predictions, **metric_kwargs)
    return metric(test_data.sentiment_data(), predictions, **metric_kwargs)


def evaluate_model(metric: Union[Callable[[np.ndarray, np.ndarray], float],
                                 Callable[['bella.data_types.TargetCollection',
                                           np.ndarray], float]],
                   dataset_name_data: Dict[str, 'bella.data_types.TargetCollection'],
                   dataset_predictions: Dict[str, np.ndarray],
                   custom_metric: bool = False,
                   **metric_kwargs) -> Dict[str, float]:
    '''
    Given a metric, dictionary of dataset names with test data and predictions
    will evaluate the predictions for each dataset.

    :param metric: Metric used to evaluate the predictions e.g.
                   :py:func:`sklearn.metrics.f1_score` or a custom function
                   like :py:func:`bella.evaluation.distinct_sentiment_metrics`
    :param dataset_name_data: Dictionary of dataset names as keys and the
                              test data as values.
    :param dataset_predictions: Dictionary of dataset names as keys and the
                                predictions as values.
    :param custom_metric: If the metric being used has come from the Bella
                          package then it is custom and this should be True
                          else use the Default False.
    :param metric_kwargs: Keyword arguments to the metric function.
    :return: Dictionary of dataset names as keys and the metric score as
             values.
    '''
    dataset_score = {}
    for dataset_name, test_data in dataset_name_data.items():
        predictions = dataset_predictions[dataset_name]
        dataset_score[dataset_name] = score(metric, test_data, predictions,
                                            custom_metric, **metric_kwargs)
    return dataset_score


def evaluate_models(metric: Union[Callable[[np.ndarray, np.ndarray], float],
                                  Callable[['bella.data_types.TargetCollection',
                                            np.ndarray], float]],
                    dataset_name_data: Dict[str, 'bella.data_types.TargetCollection'],
                    model_dataset_predictions: Dict[str,
                                                    Dict[str, np.ndarray]],
                    custom_metric: bool = False, dataframe: bool = False,
                    **metric_kwargs) -> Union[Dict[str, Dict[str, float]],
                                              'pd.DataFrame']:
    '''
    Given a metric, dictionary of dataset names with test data and
    a dictionary of models and their predictions will evaluate the predictions
    for each dataset and for each model.

    :param metric: Metric used to evaluate the predictions e.g.
                   :py:func:`sklearn.metrics.f1_score` or a custom function
                   like :py:func:`bella.evaluation.distinct_sentiment_metrics`
    :param dataset_name_data: Dictionary of dataset names as keys and the
                              test data as values.
    :param model_dataset_predictions: Dictionary of dictionaries, outer
                                      dictionary is model name and inner
                                      dictionary is dataset names as keys and
                                      predictions as values.
    :param custom_metric: If the metric being used has come from the Bella
                          package then it is custom and this should be True
                          else use the Default False.
    :param dataframe: Whether to return the results as a
                      :py:class:`pandas.DataFrame`. Default False.
    :param metric_kwargs: Keyword arguments to the metric function.
    :return: Dictionary of dictionaries, outer dictionary is the model names,
             the inner is dataset names as keys and predictions as values. If
             `dataframe` argument is True then returns it as a
             :py:class:`pandas.DataFrame` object.
    '''
    model_dataset_score = defaultdict(lambda: dict())
    for model_name, dataset_predictions in model_dataset_predictions.items():
        for dataset_name, predictions in dataset_predictions.items():
            test_data = dataset_name_data[dataset_name]
            result = score(metric, test_data, predictions,
                           custom_metric, **metric_kwargs)
            model_dataset_score[model_name][dataset_name] = result
    if dataframe:
        model_dataset_score = pd.DataFrame(model_dataset_score)
        mean_values = model_dataset_score.mean()
        mean_df = mean_values.to_frame('Mean').T
        return pd.concat((model_dataset_score, mean_df))
    return model_dataset_score


def distinct_sentiment_metrics(test_data: 'bella.data_types.TargetCollection',
                               predictions: np.ndarray,
                               distinct_sentiments: int,
                               metric_func: Callable[[np.ndarray, np.ndarray],
                                                     float],
                               **metric_kwargs) -> float:
    '''
    Custom metric function. It performs the `metric_func` on a subset of the
    test data where the samples have to come from sentences that contain
    a certain number of distinct_sentiments.

    If a sentence has two samples where each target sample is positive this
    would mean that sentence only has one distinct_sentiment. If those two
    samples contained different sentiments then that sentence has two
    distinct_sentiments. This function evaluates predictions on samples that
    have come from sentences with a defined number of distinct_sentiments
    where the evaluation function is the `metric_func` argument over the
    subset of samples based on distinct_sentiments.

    :param test_data: The test data stored as a
                      :py:class:`bella.data_types.TargetCollection` object.
    :param predictions: Array of predictions for the test data
    :param distinct_sentiments: The number of unique sentiments in a sentence.
    :param metric_func: Metric used to evaluate the predictions e.g.
                        :py:func:`sklearn.metrics.f1_score`
    :param metric_kwargs: Keyword arguments to the metric function.
    '''
    test_data_copy = copy.deepcopy(test_data)
    test_data_copy.add_pred_sentiment(predictions)
    test_subset = test_data_copy.subset_by_sentiment(distinct_sentiments)
    true_values = test_subset.sentiment_data()
    subset_predictions = test_subset.sentiment_data(sentiment_field='predicted')
    return metric_func(true_values, subset_predictions, **metric_kwargs)
