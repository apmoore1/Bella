'''
Functions that given a TargetCollection and any number of kwargs will 
return a list of `target_id`s from the given TargetCollection.

These functions are used to subset TargetCollection so that we can get 
subset metrics e.g. Accuracy score for samples within the TargetCollection 
that have only one sentiment within the sentence that sample came from
'''
from collections import defaultdict
from typing import Dict, Any, Set, List

from bella.data_types import TargetCollection, Target

def targets_to_samples(dataset: TargetCollection, targets: Set[str], 
                       lower: bool = True) -> List[Target]:
    '''
    Given a dataset and a set of target words, it will return a subset of the 
    dataset where all samples in the subset have target words that are in the 
    targets set.

    :param dataset: TargetCollection containing samples
    :param targets: A set of target words used to subset the dataset
    :param lower: Whether to lower case the target words. If this is True 
                  it is up to you to ensure all the words in the `targets` set 
                  have been lower cased.
    :returns: A subset of the dataset where all targets in the subset are 
              within the `targets` set.
    '''
    samples = []
    for data in dataset.data():
        target = data['target']
        if lower:
            target = target.lower()
        if target in targets:
            samples.append(data)
    return samples

def target_sentiments(dataset: TargetCollection, 
                      lower: bool = True) -> Dict[str, Set[Any]]:
    '''
    Given a dataset will return a dictionary of targets and the sentiment 
    that has been associated to those targets.

    E.g. within the dataset that `target` `camera` may have only been seen 
    with a positive and a negative label but not neutral therefore in the 
    returned dictionary it would be {`camera`: [`positive`, `negative`]}

    :param dataset: TargetCollection containing samples
    :param lower: Whether to lower case the target words.
    :returns: A dictionary where the keys are target words and the values 
              are the sentiment values that have been associated to those 
              targets.
    '''
    targets_sentiments = defaultdict(set)
    for data in dataset.data():
        target = data['target']
        if lower:
            target = target.lower()
        targets_sentiments[target].add(data['sentiment'])
    return targets_sentiments

def same_one_sentiment(test_dataset: TargetCollection, 
                       train_dataset: TargetCollection, 
                       lower: bool = True) -> List[str]:
    '''
    Given a test and train dataset will return all of the test dataset sample 
    ids that contain targets that have only occured once in the train and test 
    sets with the same sentiment.

    :param test_dataset: Test TargetCollection
    :param train_dataset: Train TargetCollection
    :param lower: Whether to lower case the target words
    :returns: A list of sample ids from the test dataset.
    '''
    train_target_sentiments = target_sentiments(train_dataset, lower)
    test_target_sentiments = target_sentiments(test_dataset, lower)

    same_one_sentiments = set()
    for data in test_dataset.data():
        target = data['target']
        if lower:
            target = target.lower()
        if (target in train_target_sentiments and 
            target in test_target_sentiments):
            train_sentiments = train_target_sentiments[target]
            test_sentiments = test_target_sentiments[target]
            if (len(train_sentiments) == 1 and 
                len(test_sentiments) == 1):
                if train_sentiments == test_sentiments:
                    same_one_sentiments.add(target)

    same_one_samples = targets_to_samples(test_dataset, same_one_sentiments, 
                                          lower)
    same_one_ids = [sample['target_id'] for sample in same_one_samples]
    return same_one_ids

def same_multi_sentiment(test_dataset: TargetCollection,
                         train_dataset: TargetCollection,
                         lower: bool = True) -> List[str]:
    '''
    Given a test and train dataset will return all of the test dataset sample 
    ids that contain targets that have occured more than once in the train and 
    test sets with the same sentiment labels.

    :param test_dataset: Test TargetCollection
    :param train_dataset: Train TargetCollection
    :param lower: Whether to lower case the target words
    :returns: A list of sample ids from the test dataset.
    '''
    train_target_sentiments = target_sentiments(train_dataset, lower)
    test_target_sentiments = target_sentiments(test_dataset, lower)

    same_multi_sentiments = set()
    for data in test_dataset.data():
        target = data['target']
        if lower:
            target = target.lower()
        if (target in train_target_sentiments and 
            target in test_target_sentiments):
            train_sentiments = train_target_sentiments[target]
            test_sentiments = test_target_sentiments[target]
            if (len(train_sentiments) > 1 and 
                len(test_sentiments) > 1):
                if train_sentiments == test_sentiments:
                    same_multi_sentiments.add(target)
    same_multi_samples = targets_to_samples(test_dataset, same_multi_sentiments, 
                                            lower)
    same_multi_ids = [sample['target_id'] for sample in same_multi_samples]
    return same_multi_ids

def similar_sentiment(test_dataset: TargetCollection,
                      train_dataset: TargetCollection,
                      lower: bool = True) -> List[str]:
    '''
    Given a test and train dataset will return all of the test dataset sample 
    ids that contain targets that have occured more than once in the train or 
    test sets with at least some overlap between the test sentiment and train 
    but not identical. E.g. the target `camera` could occur with `positive` and
    `negative` sentiment in the test set and only `negative` in the train set.

    :param test_dataset: Test TargetCollection
    :param train_dataset: Train TargetCollection
    :param lower: Whether to lower case the target words
    :returns: A list of sample ids from the test dataset.
    '''
    train_target_sentiments = target_sentiments(train_dataset, lower)
    test_target_sentiments = target_sentiments(test_dataset, lower)

    similar_sentiments = set()
    for data in test_dataset.data():
        target = data['target']
        if lower:
            target = target.lower()
        if (target in train_target_sentiments and 
            target in test_target_sentiments):
            train_sentiments = train_target_sentiments[target]
            test_sentiments = test_target_sentiments[target]
            if (len(train_sentiments) > 1 or 
                len(test_sentiments) > 1):
                if train_sentiments == test_sentiments:
                    continue
                if test_sentiments.intersection(train_sentiments):
                    similar_sentiments.add(target)
    similar_samples = targets_to_samples(test_dataset, similar_sentiments, 
                                         lower)
    similar_ids = [sample['target_id'] for sample in similar_samples]
    return similar_ids

def different_sentiment(test_dataset: TargetCollection,
                        train_dataset: TargetCollection, 
                        lower: bool = True) -> List[str]:
    '''
    Given a test and train dataset will return all of the test dataset sample 
    ids that contain targets that have different sentiment labels with no 
    overlap in the test compared to the train set.

    :param test_dataset: Test TargetCollection
    :param train_dataset: Train TargetCollection
    :param lower: Whether to lower case the target words
    :returns: A list of sample ids from the test dataset.
    '''
    train_target_sentiments = target_sentiments(train_dataset, lower)
    test_target_sentiments = target_sentiments(test_dataset, lower)

    different_sentiments = set()
    for data in test_dataset.data():
        target = data['target']
        if lower:
            target = target.lower()
        if (target in train_target_sentiments and 
            target in test_target_sentiments):
            train_sentiments = train_target_sentiments[target]
            test_sentiments = test_target_sentiments[target]
            if not test_sentiments.intersection(train_sentiments):
                different_sentiments.add(target)
    different_samples = targets_to_samples(test_dataset, different_sentiments, 
                                           lower)
    different_ids = [sample['target_id'] for sample in different_samples]
    return different_ids

def unknown_targets(test_dataset: TargetCollection,
                    train_dataset: TargetCollection, 
                    lower: bool = True) -> List[str]:
    '''
    Given a test and train dataset will return all of the test dataset sample 
    ids that contain targets that did not exist in the training data.

    :param test_dataset: Test TargetCollection
    :param train_dataset: Train TargetCollection
    :param lower: Whether to lower case the target words
    :returns: A list of sample ids from the test dataset.
    '''
    train_target_sentiments = target_sentiments(train_dataset, lower)
    test_target_sentiments = target_sentiments(test_dataset, lower)

    unknowns = set()
    for data in test_dataset.data():
        target = data['target']
        if lower:
            target = target.lower()
        if (target in train_target_sentiments and 
            target in test_target_sentiments):
            continue
        else:
            unknowns.add(target)

    unknown_samples = targets_to_samples(test_dataset, unknowns, lower)
    unknown_ids = [sample['target_id'] for sample in unknown_samples]
    return unknown_ids