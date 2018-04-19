from collections import defaultdict
import json
import os

import pandas as pd
from sklearn.metrics import f1_score, recall_score, accuracy_score

###
## ALL OF THIS NEEDS TESTING
###
###
## This can be re-factored so that only the raw data is stored as the the
## scores such as F1 coming from it can be inferred from the raw data easily.
###

def _raw_data_file(file_path, dataset_name):
    '''
    Given a file name to the results TSV file storing the dataset scores
    and the dataset name returns a file path which will store the raw results

    :param file_path: Path to a results file e.g. /home/user/results/tdparse.tsv
    :param dataset_name: Name of the dataset e.g. SemEval 14 Laptops.
    :type file_path: String
    :type dataset_name: String
    :returns: The file path to the raw data \
    e.g. /home/user/results/tdparse_raw_data/dataset_name.json
    :rtype: String
    '''
    # Create raw data directory
    model_name = os.path.splitext(os.path.basename(file_path))[0]
    raw_data_dir = '{}_raw_data'.format(model_name)
    raw_data_dir_path = os.path.join(os.path.dirname(file_path), raw_data_dir)
    os.makedirs(raw_data_dir_path, exist_ok=True)
    # Raw file path
    file_name = '{}.json'.format(dataset_name)
    raw_file_path = os.path.join(raw_data_dir_path, file_name)
    return raw_file_path

def get_raw_data(file_path, dataset_name, test_data):
    '''
    :param file_path: File path to the results file \
    e.g. /home/user/results/tdparse.tsv
    :param dataset_name: Name of the dataset e.g. SemEval 14 Laptops
    :param test_data: The dataset loaded into a TargetCollection
    :type file_path: String
    :type dataset_name: String
    :type test_data: TargetCollection
    :returns: A TargetCollection of the test_data but with the predicted values \
    set in the TargetCollection which comes from the raw data.
    :rtype: TargetCollection
    '''
    raw_file_path = _raw_data_file(file_path, dataset_name)
    if os.path.isfile(raw_file_path):
        with open(raw_file_path, 'r') as raw_file:
            id_pred_sent = json.load(raw_file)
            test_ids = id_pred_sent['test_id']
            predicted_sentiment = id_pred_sent['y_pred']
            ids_sent = list(zip(test_ids, predicted_sentiment))
            if len(ids_sent) != len(test_data):
                raise ValueError('The number of predicted sentiments stored {} '\
                                 'is not the same as the number of Targets in '\
                                 'the test data given {}'\
                                 .format(len(id_pred_sent), len(test_data)))
            for test_id, predicted_sentiment in ids_sent:
                test_data[test_id]['predicted'] = predicted_sentiment
            return test_data
    else:
        print('Raw file {} does not exist for dataset {}'\
        .format(raw_file_path, dataset_name))
    return None

def save_results(file_path, results_dataframe, dataset_name, test_ids=None,
                 y_pred=None, save_raw_data=False, re_write=True):
    '''
    :param file_path: File path to the results file \
    e.g. /home/user/results/tdparse.tsv
    :param results_dataframe: results to save to the file_path
    :param dataset_name: Name of the dataset e.g. SemEval 14 Laptops
    :param test_ids: ids of the test data that are associated to the y_preds
    :param y_pred: predicted labels
    :param save_raw_data: Wether to save the data as raw which is the test ids \
    and associated predicted labels
    :param re_write: If to re-write the data saved to the file name and raw data
    :type file_path: String
    :type results_dataframe: pandas.DataFrame
    :type dataset_name: String
    :type test_ids: list. Default None
    :type y_pred: list. Default None
    :type save_raw_data: bool. Default False
    :type re_write: bool. Default False
    :returns: Nothing but saves the results data to the file name and if the \
    test_ids, y_pred and save_raw_data is True will save the predicted labels \
    and associated ids to json file which then can be retrieved as a \
    TargetCollection values using get_raw_data method.
    :rtype: None
    '''
    if os.path.isfile(file_path):
        with open(file_path, 'r') as result_file:
            previous_results = pd.read_csv(result_file, sep='\t')
            results_dataframe = pd.concat([previous_results, results_dataframe])
    # Ensure the results directgory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    results_exist = get_results(file_path, dataset_name)
    if results_exist is not None:
        if re_write:
            print('Re-writing over previous results')
            with open(file_path, 'w') as result_file:
                results_dataframe.to_csv(result_file, sep='\t')
        else:
            print('Not saving results as we do not have permission to '\
                  're-write results')
    else:
        with open(file_path, 'w') as result_file:
            results_dataframe.to_csv(result_file, sep='\t')



    if save_raw_data:
        if test_ids is None or y_pred is None or dataset_name is None:
            raise ValueError('y_pred, test_ids, and dataset_name parameters have '\
                             'to be set if you would like to save the raw data')
        raw_file_path = _raw_data_file(file_path, dataset_name)
        if os.path.isfile(raw_file_path):
            if re_write:
                print('Re-writing over previous RAW results')
                with open(raw_file_path, 'w') as raw_file:
                    json.dump({'test_id' : test_ids, 'y_pred' : y_pred},
                              raw_file)
            else:
                print('Not saving RAW results as we do not have permission to '\
                      're-write results')
        else:
            with open(raw_file_path, 'w') as raw_file:
                json.dump({'test_id' : test_ids, 'y_pred' : y_pred}, raw_file)

def get_results(file_name, dataset_name):
    '''
    :param file_name: File path to the results file \
    e.g. /home/user/results/tdparse.tsv
    :param dataset_name: Name of the dataset e.g. SemEval 14 Laptops
    :type file_name: String
    :type dataset_name: String
    :returns: The results data as a Pandas DataFrame
    :rtype: pandas.DataFrame
    '''
    if os.path.isfile(file_name):
        results_dataframe = pd.read_csv(file_name, '\t')
        results_dataframe = results_dataframe.set_index('dataset')
        if dataset_name in results_dataframe.index:
            #import code
            #code.interact(local=locals())
            dataset_results = results_dataframe.loc[dataset_name]
            method_name, _ = os.path.splitext(os.path.basename(file_name))
            #dataset_results.columns = [method_name]
            return dataset_results
    return None

def scores(y_true, y_pred, num_classes):
    '''
    :param y_true: Correct labels
    :param y_pred: Predicted labels
    :param num_classes: Number of classes in the labels e.g. 2 for binary
    :type y_true: list
    :type y_pred: list
    :type num_classes: int
    :returns; A list of the following scores in this order: 1. Accuracy, \
    2. 3-way Macro F1, 3. 2-way Macro F1 and 4. Macro recall
    :rtype: list
    '''
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
    '''
    :param dataset_name: Name of the dataset e.g. SemEval 14 Laptops. NOTE \
    has to related to the dataset name in the saved results as it uses the \
    get_results method.
    :param *args: All the file paths to the stored results files of all the \
    results you would like to compare
    :returns: All of the results concatenated
    :rtype: pandas.DataFrame
    '''
    all_results = [get_results(result_file, dataset_name) for result_file in args]
    return pd.concat(all_results, axis=1)

def evaluation_results(y_pred, test, dataset_name, file_name=None,
                       save_raw_data=False, re_write=True):
    '''
    param y_pred: List of predicted scores for the test dataset in the `test` \
    param.
    :param test: TargetCollection of the test data
    :param dataset_name: Name of the test dataset. Does not have to be the real \
    name just something meaningful
    :param file_name: File path to store the results e.g. accuracy, F1 etc score \
    results. Full paths required.
    :save_raw_data: Wether to save the raw results as in the predicted sentiment \
    score per test instance.
    :type y_pred: list
    :type test: TargetCollection
    :type dataset_name: String
    :type file_name: String Optional Default None
    :type save_raw_data: bool Optional Default False
    :returns: Results dataframe where the dataframe contains results statistics \
    from the scores function which includes Accuracy, F1 3-way, F1 2-way, and \
    recall all F1 and recall are macro values. These statistics are done on the \
    whole test data as well on subsets which are per distinct sentiment values \
    per target text.
    :rtype: pandas.DataFrame
    '''

    # Check if the data already exists
    if file_name is not None and not re_write:
        # Check if the results have been saved previously
        results_df = get_results(file_name, dataset_name)
        if results_df:
            return results_df

    if not isinstance(y_pred, list):
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
            if len(sub_test_collection) == 0:
                all_test_scores.extend([0, 0, 0, 0])
            else:
                sub_y_true = sub_test_collection.sentiment_data()
                sub_y_pred = sub_test_collection.\
                             sentiment_data(sentiment_field='predicted')
                subset_num_classes = len(sub_test_collection.stored_sentiments())
                all_test_scores.extend(scores(sub_y_true, sub_y_pred,
                                              subset_num_classes))

        acc_text = 'Accuracy for text with {} distinct sentiments'.format(i)
        f1_3_text = '3 Class Macro F1 for text with {} distinct sentiments'.format(i)
        f1_2_text = '2 Class Macro F1 for text with {} distinct sentiments'.format(i)
        recall_text = 'Macro Recall for text with {} distinct sentiments'.format(i)
        results_dict[acc_text].append(all_test_scores[0])
        results_dict[f1_3_text].append(all_test_scores[1])
        results_dict[f1_2_text].append(all_test_scores[2])
        results_dict[recall_text].append(all_test_scores[3])
    results_dict['dataset'] = dataset_name
    results_df = pd.DataFrame(results_dict)
    if file_name is not None:
        # Whether to save the raw data or not
        if save_raw_data:
            print('saving raw data')
            test_ids = list(test.keys())
            save_results(file_name, results_df, dataset_name, test_ids=test_ids,
                         y_pred=y_pred, save_raw_data=save_raw_data,
                         re_write=re_write)
        else:
            save_results(file_name, results_df, dataset_name, re_write=re_write)
    return results_df
