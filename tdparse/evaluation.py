from collections import defaultdict
import json
import os

import pandas as pd
from sklearn.metrics import f1_score, recall_score, accuracy_score

###
## ALL OF THIS NEEDS TESTING
###

def _raw_data_file(file_path, dataset_name):
    '''
    :param file_path: Path to a results file e.g. /home/user/results/tdparse.tsv
    :param dataset_name: Name of the 
    '''
    # Create raw data directory
    model_name = os.path.splitext(os.path.basename(file_path))[0]
    raw_data_dir = '{}_raw_data'.format(model_name)
    raw_data_dir_path = os.path.join(os.path.dirname(file_path), raw_data_dir)
    print('Raw data dir before os makedirs {}'.format(raw_data_dir_path))
    os.makedirs(raw_data_dir_path, exist_ok=True)
    # Raw file path
    file_name = '{}.json'.format(dataset_name)
    raw_file_path = os.path.join(raw_data_dir_path, file_name)
    return raw_file_path

def get_raw_data(file_path, dataset_name):
    raw_file_path = _raw_data_file(file_path, dataset_name)
    if os.path.isfile(raw_file_path):
        with open(raw_file_path, 'r') as raw_file:
            return json.load(raw_file)
    else:
        print('Raw file {} does not exist for dataset {}'\
        .format(raw_file_path, dataset_name))

def save_results(file_path, results_dataframe, y_true=None, y_pred=None,
                 dataset_name=None, save_raw_data=False):
    if os.path.isfile(file_path):
        with open(file_path, 'r') as result_file:
            previous_results = pd.read_csv(result_file, sep='\t')
            results_dataframe = pd.concat([previous_results, results_dataframe])
    # Ensure the results directgory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as result_file:
        results_dataframe.to_csv(result_file, sep='\t')
    print('raw data {}'.format(save_raw_data))
    if save_raw_data:
        if y_true is None or y_pred is None or dataset_name is None:
            raise ValueError('y_pred, y_true, and dataset_name parameters have '\
                             'to be set if you would like to save the raw data')
        raw_file_path = _raw_data_file(file_path, dataset_name)
        print('raw file path {}'.format(raw_file_path))
        # Saving data in json format
        with open(raw_file_path, 'w') as raw_file:
            json.dump({'y_ture' : y_true, 'y_pred' : y_pred}, raw_file)

def get_results(file_name, dataset_name):
    if os.path.isfile(file_name):
        results_dataframe = pd.read_csv(file_name, '\t')
        if dataset_name in results_dataframe.index:
            dataset_results = results_dataframe.loc[dataset_name]
            method_name, _ = os.path.splitext(os.path.basename(file_name))
            dataset_results.columns = [method_name]
            return dataset_results
    return False

def scores(y_true, y_pred, num_classes):
    norm_score = lambda score: round(score * 100, 1)
    acc = norm_score(accuracy_score(y_true, y_pred))
    f1 = 0
    f1_2way = 0
    if num_classes == 2:
        f1_2way = norm_score(f1_score(y_true, y_pred, average='macro'))
    elif num_classes == 3:
        f1 = norm_score(f1_score(y_true, y_pred, average='macro'))
        f1_2way = norm_score(f1_score(y_true, y_pred, average='macro',
                                      labels=[-1, 1]))
    macro_recall = norm_score(recall_score(y_true, y_pred, average='macro'))
    return [acc, f1, f1_2way, macro_recall]

def combine_results(dataset_name, *args):
    all_results = [get_results(result_file, dataset_name) for result_file in args]
    return pd.concat(all_results, axis=1)

def evaluation_results(y_pred, test, dataset_name, file_name=None,
                       save_raw_data=False):
    y_pred = y_pred.tolist()
    y_true = test.sentiment_data()
    test.add_pred_sentiment(y_pred)

    num_classes = len(test.stored_sentiments())
    test_scores = scores(y_true, y_pred, num_classes)

    results_dict = defaultdict(list)
    results_dict['Accuracy'].append(test_scores[0])
    results_dict['3 Class Macro F1'].append(test_scores[1])
    results_dict['2 Class F1'].append(test_scores[2])
    results_dict['Macro Recall'].append(test_scores[3])

    # Split the test set into sentences that contain i unique sentiments
    for i in range(1, 4):
        all_test_scores = []
        if i > num_classes:
            all_test_scores.extend([0, 0, 0, 0])
        else:
            sub_test_collection = test.subset_by_sentiment(i)
            sub_y_true = sub_test_collection.sentiment_data()
            sub_y_pred = sub_test_collection.sentiment_data(sentiment_field='predicted')
            subset_num_classes = len(sub_test_collection.stored_sentiments())
            all_test_scores.extend(scores(sub_y_true, sub_y_pred, subset_num_classes))

        acc_text = 'Accuracy for text with {} distinct sentiments'.format(i)
        f1_3_text = '3 Class Macro F1 for text with {} distinct sentiments'.format(i)
        f1_2_text = '2 Class Macro F1 for text with {} distinct sentiments'.format(i)
        recall_text = 'Macro Recall for text with {} distinct sentiments'.format(i)
        results_dict[acc_text].append(all_test_scores[0])
        results_dict[f1_3_text].append(all_test_scores[1])
        results_dict[f1_2_text].append(all_test_scores[2])
        results_dict[recall_text].append(all_test_scores[3])
    results_df = pd.DataFrame(results_dict, index=[dataset_name])
    if file_name is not None:
        # Whether to save the raw data or not
        print('file name')
        if save_raw_data:
            print('saving raw data')
            save_results(file_name, results_df, y_true=y_true, y_pred=y_pred,
                         save_raw_data=save_raw_data, dataset_name=dataset_name)
        else:
            save_results(file_name, results_df)
    return results_df
