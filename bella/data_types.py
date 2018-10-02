'''
Module that contains the various data types:

1. Target -- Mutable data store for a single Target value i.e. one training \
example.
2. TargetCollection -- Mutable data store for Target data types.  i.e. A \
data store that contains multiple Target instances.
'''

from collections.abc import MutableMapping
from collections import OrderedDict, defaultdict
import copy
import json
from pathlib import Path
from typing import List, Callable, Union, Dict, Any

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns

from bella.tokenisers import whitespace
from bella.stanford_tools import constituency_parse

BELLA_DATASET_DIR: Path = Path.home().joinpath('.Bella', 'Datasets')

class Target(MutableMapping):
    '''
    Mutable data store for a single Target value. This should be used as the
    value for Target information where it contains all data required to be
    classified as Target data for Target based sentiment classification.

    Overrides collections.abs.MutableMapping abstract class.

    Reference on how I created the class:
    http://www.kr41.net/2016/03-23-dont_inherit_python_builtin_dict_type.html
    https://docs.python.org/3/library/collections.abc.html

    Functions changed compared to normal:

    1. __delitem__ -- Will only delete the `target_id` key.
    2. __eq__ -- Two Targets are the same if they either have the same `id` or \
    they have the same values to the minimum keys \
    ['spans', 'text', 'target', 'sentiment']
    3. __setitem__ -- Only allows you to add/modify the `predicted` key which \
    represents the predicted sentiment for the Target instance.
    '''

    def __init__(self, spans, target_id, target, text, sentiment, predicted=None,
                 sentence_id=None):
        '''
        :param target: Target that the sentiment is about. e.g. Iphone
        :param sentiment: sentiment of the target.
        :param text: The text context that the target and sentiment is within.
        :param target_id: Unique ID. Has to be Unique within the \
        TargetCollection if it is put into a TargetCollection.
        :param spans: List of tuples where each tuple is of length 2 where they \
        contain the exclusive range of an instance of Target word in the Text \
        context. The reason it is a list if because the Target word can be \
        mentioned more than once e.g. `The Iphone was great but the iphone is \
        small`. The first Int in the tuple has to be less than the second Int.
        :param predicted: If given adds the predicted sentiment value.
        :param sentence_id: Unique ID of the sentence that the target is \
        within. More than one target can have the same sentence.
        :type target: String
        :type sentiment: String or Int (Based on annotation schema)
        :type text: String
        :type target_id: String
        :type spans: list
        :type predicted: Same type as sentiment. Default None (Optional)
        :type sentence_id: String. Default None (Optional)
        :returns: Nothing. Constructor.
        :rtype: None
        '''
        if not isinstance(target_id, str):
            raise TypeError('The target ID has to be of type String and not {}'\
                            .format(type(target_id)))
        if not isinstance(target, str):
            raise TypeError('The target has to be of type String and not {}'\
                            .format(type(target)))
        if not isinstance(text, str):
            raise TypeError('The text has to be of type String and not {}'\
                            .format(type(text)))
        if not isinstance(sentiment, (str, int)):
            raise TypeError('The sentiment has to be of type String or Int and '\
                            'not {}'.format(type(sentiment)))
        if not isinstance(spans, list):
            raise TypeError('The spans has to be of type list and not {}'\
                            .format(type(spans)))
        else:
            if len(spans) < 1:
                raise TypeError('spans has to contain at least one tuple not '\
                                'None')
            else:
                for span in spans:
                    if not isinstance(span, tuple):
                        raise TypeError('Spans has to be a list of tuples not {}'\
                                        .format(type(span)))
                    if len(span) != 2:
                        raise ValueError('Spans must contain tuples of length'\
                                         ' 2 not {}'.format(spans))
                    if not isinstance(span[0], int) or \
                       not isinstance(span[1], int):
                        raise TypeError('spans must be made of tuple containing '\
                                        'two Integers not {}'.format(span))
                    if span[1] <= span[0]:
                        raise ValueError('The first integer in a span must be '\
                                         'less than the second integer {}'\
                                         .format(span))
        temp_dict = dict(spans=spans, target_id=target_id, target=target,
                         text=text, sentiment=sentiment)
        if sentence_id is not None:
            if not isinstance(sentence_id, str):
                raise TypeError('`sentence_id` has to be a String and not {}'\
                                .format(type(sentence_id)))
            temp_dict['sentence_id'] = sentence_id

        self._storage = temp_dict
        if predicted is not None:
            self['predicted'] = predicted


    def __getitem__(self, key):
        return self._storage[key]

    def __iter__(self):
        return iter(self._storage)

    def __len__(self):
        return len(self._storage)

    def __delitem__(self, key):
        '''
        To ensure that the Target class maintains the minimum Keys and Values
        to allow an instance to be used in Target based machine learning. The
        key and associated values that can be deleted are limited to:

        1. target_id

        :param key: The key and associated value to delete from the store.
        :returns: Updates the data store by removing key and value.
        :rtype: None
        '''

        accepted_keys = set(['target_id'])
        if key not in accepted_keys:
            raise KeyError('The only keys that can be deleted are the '\
                           'following: {} the key you wish to delete {}'\
                           .format(accepted_keys, key))
        del self._storage[key]

    def __setitem__(self, key, value):
        '''
        :param key: key (Only store values for `predicted` key)
        :param value: Predicted sentiment value which has to be the same data \
        type as the `sentiment` value.
        :type key: String (`predicted` is the only key accepted at the moment)
        :type value: Int or String.
        :returns: Nothing. Adds the predicted sentiment of the Target.
        :rtype: None.
        '''

        if key != 'predicted':
            raise KeyError('The Only key that can be changed is the `predicted`'\
                           ' key not {}'.format(key))
        #raise_type = False
        #sent_value = self._storage['sentiment']
        #if isinstance(sent_value, int):
        #    if not isinstance(value, (int, np.int32, np.int64)):
        #        raise_type = True
        #elif not isinstance(value, type(sent_value)):
        #    raise_type = True

        #if raise_type:
        #    raise TypeError('Value to be stored for the `predicted` sentiment '\
        #                    'has to be the same data type as the sentiment '\
        #                    'value {} and not {}.'\
        #                    .format(sent_value, type(value)))
        self._storage[key] = value

    def __repr__(self):
        '''
        :returns: String returned is what user see when the instance is \
        printed or printed within a interpreter.
        :rtype: String
        '''

        return 'Target({})'.format(self._storage)

    def __eq__(self, other):
        '''
        Two Target instances are equal if they are both Target instances and
        one of the following conditions

        1. They have the same target_id (This is preferred)
        2. The minimum keys that all targets have to have \
        ['spans', 'text', 'target', 'sentiment'] are all equal.

        :param other: The target instance that is being compare to the current \
        target instance.
        :type other: Target
        :returns: True if they are equal else False.
        :rtype: bool
        '''

        if not isinstance(other, Target):
            return False
        if 'target_id' in self and 'target_id' in other:
            if self['target_id'] != other['target_id']:
                return False
        else:
            minimum_keys = ['spans', 'text', 'target', 'sentiment']
            for key in minimum_keys:
                if not self[key] == other[key]:
                    return False
        return True
    def __array__(self):
        '''
        Function for converting it to a numpy array
        '''
        return np.asarray(dict(self))

class TargetCollection(MutableMapping):
    '''
    Mutable data store for Target data types.  i.e. A data store that contains
    multiple Target instances. This collection ensures that there are no two
    Target instances stored that have the same ID this is because the storage
    of the collection is an OrderedDict.

    Overrides collections.abs.MutableMapping abstract class.

    Functions:

    1. add -- Given a Target instance with an `id` key adds it to the data 
       store.
    2. data -- Returns all of the Target instances stored as a list of Target 
       instances.
    3. stored_sentiments -- Returns a set of unique sentiments stored.
    4. sentiment_data -- Returns the list of all sentiment values stored in 
       the Target instances stored.
    5. add_pred_sentiment -- Adds a list of predicted sentiment values to the 
       Target instances stored.
    6. confusion_matrix -- Returns the confusion matrix between the True and 
       predicted sentiment values.
    7. subset_by_sentiment -- Creates a new TargetCollection based on the 
       number of unique sentiments in a sentence.
    8. to_json_file -- Returns a Path to a json file that has stored the 
                       data as a sample json encoded per line. If the split 
                       argument is set will return two Paths the first being 
                       the training file and the second the test.
    '''

    def __init__(self, list_of_target=None):
        '''
        :param list_of_target: An interator of Target instances e.g. a List of 
                               Target instances.
        :type list_of_target: Iterable. Default None (Optional)
        :returns: Nothing. Constructor.
        :rtype: None
        '''

        self._storage = OrderedDict()
        if list_of_target is not None:
            if not hasattr(list_of_target, '__iter__'):
                raise TypeError('The list_of_target argument has to be iterable')
            for target in list_of_target:
                self.add(target)

    def __getitem__(self, key):
        return self._storage[key]

    def __setitem__(self, key, value):
        '''
        If key already exists will raise KeyError.

        :param key: key that stores the index to the value
        :param value: value to store at the keys location
        :type key: hashable object
        :type value: Target
        :returns: Nothing. Adds data to the collection
        :rtype: None.
        '''

        if not isinstance(value, Target):
            raise TypeError('All values in this store have to be of type '\
                            'Target not {}'.format(type(value)))
        if key in self._storage:
            raise KeyError('This key: `{}` already exists with value `{}` '\
                           'value that for the same key is `{}`'\
                           .format(key, self._storage[key], value))
        temp_value = copy.deepcopy(value)
        # As the id will be saved as the key no longer needed in the target
        # instance (value). However if the key does not match the `target_id`
        # raise KeyError
        if 'target_id' in value:
            if value['target_id'] != key:
                raise KeyError('Cannot add this to the data store as the key {}'\
                               ' is not the same as the `target_id` in the Target'\
                               ' instance value {}'.format(key, value))
            del temp_value['target_id']
        self._storage[key] = temp_value

    def __delitem__(self, key):
        del self._storage[key]

    def __iter__(self):
        return iter(self._storage)

    def __len__(self):
        return len(self._storage)

    def add(self, value):
        '''
        Adds the Target instance to the data store without having to extract
        out the target_id of the target if using __setitem__

        :Example:

        >>> target = Target([(10, 16)], '1', 'Iphone',
                            'text with Iphone', 0)
        >>> target_col = TargetCollection()
        # Add method is simpler to use than __setitem__
        >>> target_col.add(target)
        # Example of the __setitem__ method
        >>> target_col[target['target_id']] = target

        :param value: Target instance with a `target_id` key
        :type value: Target
        :returns: Nothing. Adds the target instance to the data store.
        :rtype: None
        '''

        if not isinstance(value, Target):
            raise TypeError('All values in this store have to be of type '\
                            'Target not {}'.format(type(value)))
        if 'target_id' not in value:
            raise ValueError('The Target instance given {} does not have a '\
                             'target_id'.format(value))
        self[value['target_id']] = value

    def data(self):
        '''
        :returns: a list of all the Target instances stored.
        :rtype: list
        '''

        _data = []
        for _id, target_data in self.items():
            data_dict = {**target_data}
            data_dict['target_id'] = _id
            _data.append(Target(**data_dict))
        return _data

    def data_dict(self):
        '''
        :returns: Same as the data function but returns dicts instead of \
        Targets
        :rtype: list
        '''

        _data = []
        for _id, target_data in self.items():
            data_dict = {**target_data}
            data_dict['target_id'] = _id
            _data.append(data_dict)
        return _data

    def stored_sentiments(self):
        '''
        :returns: A set of all unique sentiment values of the target instances \
        in the data store.
        :rtype: set
        '''

        unique_sentiments = set()
        for target_data in self.values():
            unique_sentiments.add(target_data['sentiment'])
        return unique_sentiments

    def sentiment_data(self, mapper=None, sentiment_field='sentiment'):
        '''
        :param mapper: A dictionary that maps the keys to the values where the \
        keys are the current unique sentiment values of the target instances \
        stored
        :param sentiment_field: Determines if it should return True sentiment \
        of the Targets `sentiment` or to return the predicted value `predicted`
        :type mapper: dict
        :type sentiment_field: String. Default `sentiment` (True values)
        :returns: a list of the sentiment value for each Target instance stored.
        :rtype: list

        :Example of using the mapper:
        >>> target_col = TargetCollection([Target([(10, 16)], '1', 'Iphone',
                                                  'text with Iphone', 'pos')])
        >>> target_col.add(Target([(10, 15)], '2', 'Pixel',
                                  'text with Pixel', 'neg'))
        # Get the unique sentiment values for each target instance
        >>> map_keys = target_col.stored_sentiments()
        >>> print(map_keys)
        >>> ['pos', 'neg']
        >>> mapper = {'pos' : 1, 'neg' : -1}
        >>> print(target_col.sentiment_data(mapper=mapper))
        >>> 1, -1
        '''

        allowed_fields = set(['sentiment', 'predicted'])
        if sentiment_field not in allowed_fields:
            raise ValueError('The `sentiment_field` has to be one of the '\
                             'following values {} and not {}'\
                             .format(allowed_fields, sentiment_field))

        if mapper is not None:
            if not isinstance(mapper, dict):
                raise TypeError('The mapper has to be of type dict and not {}'\
                                .format(type(mapper)))
            allowed_keys = self.stored_sentiments()
            if len(mapper) != len(allowed_keys):
                raise ValueError('The mapper has to contain a mapping for each '\
                                 'unique sentiment value {} and not a subset '\
                                 'given {}'.format(allowed_keys, mapper.keys()))
            for map_key in mapper:
                if map_key not in allowed_keys:
                    raise ValueError('The mappings are not correct. The map '\
                                     'key {} does not exist in the unique '\
                                     'sentiment values in the store {}'\
                                     .format(map_key, allowed_keys))
            return [mapper[target_data[sentiment_field]]\
                    for target_data in self.values()]

        return [target_data[sentiment_field] for target_data in self.values()]

    def add_id_pred(self, id_pred):
        count = 0
        for targ_id in self:
            if targ_id in id_pred:
                self[targ_id]['predicted'] = id_pred[targ_id]
                count += 1
        if count != len(self):
            raise ValueError('We have only added {} predictions to {} targets'\
                             .format(count, len(self)))

    def add_pred_sentiment(self, sent_preds, mapper=None):
        '''
        :param sent_preds: A list of predicted sentiments for all Target \
        instances stored or a numpy array where columns are number of \
        different predicted runs and the rows represent the associated \
        Target instance.
        :param mapper: A dictionary mapping the predicted sentiment to \
        alternative values e.g. Integer values to String values.
        :type sent_preds: list or numpy array
        :type mapper: dict
        :returns: Nothing. Adds the predicted sentiments to the Target \
        instances stored.
        :rtype: None
        '''

        if len(sent_preds) != len(self):
            raise ValueError('The length of the predicted sentiments {} is not '\
                             'equal to the number Target instances stored {}'\
                             .format(len(sent_preds), len(self)))
        for index, target in enumerate(self.data()):
            predicted_sent = sent_preds[index]
            if mapper is not None:
                predicted_sent = mapper[predicted_sent]
            target_id = target['target_id']
            self._storage[target_id]['predicted'] = predicted_sent

    def confusion_matrix(self, plot=False, norm=False):
        '''
        :param plot: To return a heatmap of the confusion matrix.
        :param norm: Normalise the values in the confusion matrix
        :type plot: bool. Default False
        :type norm: bool. Default False
        :returns: A tuple of length two. 1. the confusion matrix \
        2. The plot of the confusion matrix if plot is True else \
        None.
        :rtype: tuple
        '''

        sentiment_values = sorted(self.stored_sentiments())
        true_values = self.sentiment_data()
        pred_values = self.sentiment_data(sentiment_field='predicted')
        conf_matrix = metrics.confusion_matrix(true_values, pred_values,
                                               labels=sentiment_values)
        if norm:
            conf_matrix = conf_matrix / conf_matrix.sum()
        conf_matrix = pd.DataFrame(conf_matrix, columns=sentiment_values,
                                   index=sentiment_values)
        ax = None
        if plot:
            if norm:
                ax = sns.heatmap(conf_matrix, annot=True, fmt='.2f')
            else:
                ax = sns.heatmap(conf_matrix, annot=True, fmt='d')
        return conf_matrix, ax

    def subset_by_sentiment(self, num_unique_sentiments):
        '''
        Creates a subset based on the number of unique sentiment values per
        sentence. E.g. if num_unique_sentiments = 2 then it will
        return all the Target instances where each target intance has at least
        two target instances per sentence and those targets can have only one
        of two sentiment values. This can be used to test how well a method
        can extract exact sentiment information for the associated target.

        NOTE: Requires that all Target instances stored contain a sentence_id.

        :param num_unique_sentiments: Integer specifying the number of unique \
        sentiments in the target instances per sentence.
        :type num_unique_sentiments: int
        :returns: A subset based on the number of unique sentiments per sentence.
        :rtype: TargetCollection
        '''



        all_relevent_targets = []
        for targets in self.group_by_sentence().values():
            target_col = TargetCollection(targets)
            if len(target_col.stored_sentiments()) == num_unique_sentiments:
                all_relevent_targets.extend(targets)
        return TargetCollection(all_relevent_targets)

    def subset_by_sentence_length(self, length_condition):

        all_relevent_targets = []
        for target in self.data():
            target_text = target['text']
            if length_condition(target_text):
                all_relevent_targets.append(target)
        return TargetCollection(all_relevent_targets)

    def to_json_file(self, dataset_name: Union[str, List[str]], 
                     split: Union[float, None] = None, cache: bool = True, 
                     **split_kwargs) -> Union[Path, List[Path]]:
        '''
        Returns a Path to a json file that has stored the data as a sample json 
        encoded per line. If the split argument is set will return two Paths 
        the first being the training file and the second the test.

        The Path does not need to be specified as it saves it to the 
        `~/.Bella/datasets/.` directory within your user space under the 
        dataset_name.

        If the split argument is used. NOTE that splitting is done in a 
        stratified fashion

        :param dataset_name: Name to associate to the dataset e.g. 
                             `SemEval 2014 rest train`. If split is not None 
                             then use a List of Strings e.g. 
                             [`SemEval 2014 rest train`, 
                             `SemEval 2014 rest dev`]
        :param split: Whether or not to split the dataset into train, test 
                      split. If not use None else specify the fraction of 
                      the data to use for the test split.
        :param cache: If the data is already saved use the Cache. Default 
                      is to use the cached data.
        :param split_kwargs: Keywords argument to give to the train_test_split 
                             function that is used for splitting.
        '''
        def create_json_file(fp: Path, data: List[Dict[str, Any]]) -> None:
            '''
            Given the a list of dictionaries that represent the Target data 
            converts these samples into json encoded samples which are saved on 
            each line within the file at the given file path(fp)

            :param fp: File path that will store the json samples one per line
            :param data: List of dictionaries that represent the Target data.
            :return: Nothing that data will be saved to the file.
            '''
            with fp.open('w+') as json_file:
                for index, target_data in enumerate(data):
                    json_encoded_data = json.dumps(target_data)
                    if index != 0:
                        json_encoded_data = f'\n{json_encoded_data}'
                    json_file.write(json_encoded_data)
        
        # If splitting the data there has to be two dataset names else one name
        if split is None:
            assert isinstance(dataset_name, str)
        elif isinstance(split, float):
            assert isinstance(dataset_name, list)
            assert len(dataset_name) == 2
        
        BELLA_DATASET_DIR.mkdir(parents=True, exist_ok=True)

        dataset_names = dataset_name
        if not isinstance(dataset_name, list):
            dataset_names = [dataset_name]
        all_paths_exist = True
        dataset_paths = []
        for name in dataset_names:
            dataset_path = BELLA_DATASET_DIR.joinpath(name)
            if not dataset_path.exists():
                all_paths_exist = False
            dataset_paths.append(dataset_path)
        # Caching
        if cache and all_paths_exist:
            print(f'Using cache for the follwoing datasets: {dataset_names}')
            if split is None:
                return dataset_paths[0]
            return dataset_paths

        target_data = self.data_dict()
        if split is None:
            create_json_file(dataset_paths[0], target_data)
            return dataset_paths[0]
        # Splitting
        sentiment_data = self.sentiment_data()
        X_train, X_test, _, _ = train_test_split(target_data, sentiment_data, 
                                                 stratify=sentiment_data, 
                                                 test_size=split)
        create_json_file(dataset_paths[0], X_train)
        create_json_file(dataset_paths[1], X_test)
        return dataset_paths

    # Not tested
    def targets_per_sentence(self):
        '''
        :returns: Dictionary of number of targets as keys and values the number \
        of sentences that have that many targets per sentence.
        :rtype: dict

        :Example:
        If we have 5 sentences that contain 1 target each and 4 sentences that
        contain 3 targets each then it will return a dict like:
        {1 : 5, 3 : 4}
        '''

        targets_sentence = {}
        for targets in self.group_by_sentence().values():
            num_targets = len(targets)
            targets_sentence[num_targets] = targets_sentence.get(num_targets, 0) + 1
        return targets_sentence

    # Not tested
    def avg_targets_per_sentence(self):
        return len(self) / self.number_sentences()
    # Not tested
    def number_sentences(self):
        return len(self.group_by_sentence())
    # Not tested
    def number_unique_targets(self):
        target_count = {}
        for target_instance in self.values():
            target = target_instance['target']
            target_count[target] = target_count.get(target, 0) + 1
        return len(target_count)

    def no_targets_sentiment(self):
        sentiment_targets = {}
        for target in self.values():
            sentiment = target['sentiment']
            sentiment_targets[sentiment] = sentiment_targets.get(sentiment, 0) + 1
        return sentiment_targets

    def ratio_targets_sentiment(self):
        no_sentiment_target = self.no_targets_sentiment()
        total_targets = sum(no_sentiment_target.values())
        ratio_sentiment_targets = {}
        for sentiment, no_targets in no_sentiment_target.items():
            ratio_sentiment_targets[sentiment] = round(no_targets / total_targets, 2)
        return ratio_sentiment_targets

    def group_by_sentence(self):
        '''
        :returns: A dictionary of sentence_id as keys and a list of target \
        instances that have the same sentence_id as values.
        :rtype: defaultdict (default is list)
        '''

        sentence_targets = defaultdict(list)
        for target in self.data():
            if 'sentence_id' not in target:
                raise ValueError('A Target id instance {} does not have '\
                                 'a sentence_id which is required.'\
                                 .format(target))
            sentence_id = target['sentence_id']
            sentence_targets[sentence_id].append(target)
        return sentence_targets

    def avg_constituency_depth(self):
        avg_depths = []
        for data in self.values():
            sentence_trees = constituency_parse(data['text'])
            tree_depths = [tree.height() - 1 for tree in sentence_trees]
            avg_depth = sum(tree_depths) / len(sentence_trees)
            avg_depths.append(avg_depth)
        return sum(avg_depths) / len(avg_depths)

    def avg_sentence_length_per_target(self, tokeniser=whitespace):
        all_sentence_lengths = []
        for data in self.values():
            all_sentence_lengths.append(len(tokeniser(data['text'])))
        return sum(all_sentence_lengths) / len(all_sentence_lengths)

    def word_list(self, tokeniser: Callable[[str], List[str]],
                  min_df: int = 0, lower: bool = True) -> List[str]:
        '''
        :param tokeniser: Tokeniser function to tokenise the text 
                          of each target/sample
        :param min_df: Optional. The minimum percentage of documents a 
                       token must occur in.
        :param lower: Optional. Whether to lower the text. 
        :return: A word list of all tokens that occur in this data collection 
                 given min_df.
        '''

        token_df = defaultdict(lambda: 0)
        num_df = 0
        for target in self.values():
            num_df += 1
            tokens = tokeniser(target['text'])
            for token in tokens:
                if lower:
                    token = token.lower()
                token_df[token] += 1
        min_df_value = int((num_df / 100) * min_df)
        word_list = [token for token, df in token_df.items()
                     if df > min_df_value]
        return word_list

    @staticmethod
    def combine_collections(*args):
        all_targets = []
        for collections in args:
            all_targets.extend(collections.data())
        return TargetCollection(all_targets)

    def __eq__(self, other):

        if len(self) != len(other):
            return False
        for key in self:
            if key not in other:
                return False
        return True

    def __repr__(self):
        '''
        :returns: String returned is what user see when the instance is \
        printed or printed within a interpreter.
        :rtype: String
        '''

        target_strings = ''

        self_len = len(self)
        if self_len > 2:
            for index, target in enumerate(self.data()):
                if index == 0:
                    target_strings += '{} ... '.format(target)
                if index == self_len - 1:
                    target_strings += '{}'.format(target)
        else:
            for target in self.data():
                target_strings += '{}, '.format(target)
        if target_strings != '':
            target_strings = target_strings.rstrip(', ')
        return 'TargetCollection({})'.format(target_strings)
