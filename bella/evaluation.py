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
from itertools import product
from typing import Callable, Union, Dict, List, Tuple, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import bella
from bella.data_types import TargetCollection


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

def get_kwargs(key: str, 
               potential_kwargs: Optional[Dict[str, Dict[str, Any]]] = None
               ) -> Dict[str, Any]:
    kwargs = {}
    if potential_kwargs is None:
        kwargs = {}
    elif key not in potential_kwargs:
        kwargs = {}
    else:
        kwargs = potential_kwargs[key]
    return kwargs

def plot_probability(data: pd.DataFrame, metric: str, 
                     model_name: Optional[str] = None,
                     models_to_remove: Optional[List[str]] = None,
                     bar_plot: bool = False, 
                     box_plot: bool = True,
                     cat_plot=False,
                     n_boot=100000,
                     **plot_kwargs):
    '''
    This assumes a dataframe that has the following columns:
    
    1. Metric
    2. Score
    3. Model
    4. Probability
    '''

    if model_name is not None:
        data = data[data['Model'] == model_name]
    data = data[data['Metric'] == metric]
    
    if not cat_plot:
        fig, ax = plt.subplots(1,1, **plot_kwargs)
    if box_plot:
        sns.boxplot(x='Probability', y='Score', hue='Model', 
                    data=data, ax=ax)
    elif bar_plot:
        if model_name is None:
            sns.barplot(x='Probability', y='Score', hue='Model', n_boot=n_boot,
                         data=data, ax=ax, ci=95)
        else:
            sns.barplot(x='Probability', y='Score', n_boot=n_boot,
                        data=data, ax=ax, ci=95)
    else:
        ax = sns.catplot(x='Probability', y='Score',  
                    col='Model', data=data, kind='bar',
                    n_boot=n_boot,
                    col_wrap=3, facet_kws=plot_kwargs)
        
    min_score = data['Score'].min()
    min_score = min_score - (min_score / 100)
    max_score = data['Score'].max()
    max_score = max_score + (max_score / 100)
    if not cat_plot:
        ax.set_ylim([min_score, max_score])
    else:
        ax.set(ylim=(min_score, max_score))
    return ax

def plot_acc_f1(results: pd.DataFrame, title: str, **plot_kwargs
                ) -> plt.Figure:
    fig, axs = plt.subplots(1, 2, **plot_kwargs)
    accuracy_data = results[results['Metric']=='Accuracy']
    f1_data = results[results['Metric']=='F1']
    axs[0] = sns.boxplot(x='Model', y='Score', data=accuracy_data, ax=axs[0])
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title(title)
    axs[1] = sns.boxplot(x='Model', y='Score', data=f1_data, ax=axs[1])
    axs[1].set_ylabel('F1')
    axs[1].set_title(title)
    return fig

def datasets_df(datasets: TargetCollection, 
                metrics: List[Tuple[str, 
                                    Callable[[np.ndarray, np.ndarray], 
                                             float]]],
                metric_funcs_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
                additional_data: Optional[Dict[str, Dict[str, Any]]] = None
                ) -> pd.DataFrame:
    num_datasets = len(datasets)
    dataset_names = [dataset.name for dataset in datasets]
    if len(set(dataset_names)) != num_datasets:
        raise ValueError('datasets given must have unique name these datasets'
                         f' do not: {dataset_names}')
    df_data = defaultdict(list)
    for dataset in datasets:
        dataset_name = dataset.name
        for metric_name, metric_func in metrics:
            metric_kwargs = get_kwargs(metric_name, metric_funcs_kwargs)
            scores = dataset.dataset_metric_scores(metric_func, **metric_kwargs)
            for score in scores:
                df_data['Model'].append(dataset_name)
                df_data['Metric'].append(f'{metric_name}')
                df_data['Score'].append(score)
                if additional_data is None:
                    continue    
                if dataset_name in additional_data:
                    additional_cols = additional_data[dataset_name]
                    for column_name, value in additional_cols.items():
                        df_data[column_name].append(value)
    return pd.DataFrame(df_data)

def summary_errors(datasets: TargetCollection,
                   metrics: List[Tuple[str, 
                                       Callable[[np.ndarray, np.ndarray], 
                                                float]]],
                   error_funcs: List[Tuple[str, Callable[[TargetCollection], 
                                                         List[str]]]],
                   metric_funcs_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
                   error_funcs_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
                   std_err: bool = True
                   ) -> pd.DataFrame:

    num_datasets = len(datasets)
    dataset_names = [dataset.name for dataset in datasets]
    if len(set(dataset_names)) != num_datasets:
        raise ValueError('datasets given must have unique name these datasets'
                         f' do not: {dataset_names}')

    error_names = [error_func_name for error_func_name, _ in error_funcs]
    metric_names = [metric_name for metric_name, _ in metrics]
    columns = list(product(error_names, metric_names))
    num_columns = len(columns)
    columns = pd.MultiIndex.from_tuples(columns)
    summary_data = pd.DataFrame(np.empty((num_datasets, num_columns)),
                                index=dataset_names, columns=columns)

    error_name_ids = []
    for error_func_name, error_func in error_funcs:
        kwargs = get_kwargs(error_func_name, error_funcs_kwargs)
        error_ids = error_func(datasets[0], **kwargs)
        error_name_ids.append((error_func_name, error_ids))
    for error_name, error_ids in error_name_ids:
        for dataset in datasets:
            dataset_name = dataset.name
            error_data = dataset.subset_by_ids(error_ids)
            for metric_name, metric_func in metrics:
                metric_kwargs = get_kwargs(metric_name, metric_funcs_kwargs)
                scores = error_data.dataset_metric_scores(metric_func, 
                                                          **metric_kwargs)
                mean_score = np.mean(scores)
                score = mean_score * 100
                score = f'{score:.2f}'
                if std_err:
                    std_score = np.std(scores) * 100
                    score = f'{score} ({std_score:.2f})'
                summary_data.loc[(dataset_name), 
                                 (error_name, metric_name)] = score
    return summary_data