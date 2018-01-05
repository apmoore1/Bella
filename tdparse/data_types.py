'''
Module that contains the various data types:

1. Target -- Non-Mutable data store for a single Target value i.e. one training \
example.
2. TargetCollection -- Mutable data store for Target data types.  i.e. A \
data store that contains multiple Target instances.
'''

from collections.abc import Mapping
from collections.abc import MutableMapping
from collections import OrderedDict
import copy

class Target(Mapping):
    '''
    Non-Mutable data store for a single Target value. This should be used as the
    value for Target information where it contains all data required to be
    classified as Target data for Target based sentiment classification.

    Overrides collections.abs.Mapping abstract class.

    Reference on why I picked the Mapping abstract base class:
    http://www.kr41.net/2016/03-23-dont_inherit_python_builtin_dict_type.html
    https://docs.python.org/3/library/collections.abc.html

    Added functions:

    1. __delitem__ -- Deletes a key and its associated value in the data store
    2. __eq__ -- Two Targets are the same if they either have the same `id` or \
    they have the same values to the minimum keys \
    ['spans', 'text', 'target', 'sentiment']
    '''

    def __init__(self, spans, target_id, target, text, sentiment):
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
        :type target: String
        :type sentiment: String or Int (Based on annotation schema)
        :type text: String
        :type target_id: String
        :type spans: list
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


        self._storage = dict(spans=spans, id=target_id, target=target,
                             text=text, sentiment=sentiment)

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

        1. id

        :param key: The key and associated value to delete from the store.
        :returns: Updates the data store by removing key and value.
        :rtype: None
        '''

        accepted_keys = set(['id'])
        if key != 'id':
            raise KeyError('The only keys that can be deleted are the '\
                           'following: {} the key you wish to delete {}'\
                           .format(accepted_keys, key))
        del self._storage[key]

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

        1. They have the same ID (This is preferred)
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
        if 'id' in self and 'id' in other:
            if self['id'] != other['id']:
                return False
        else:
            minimum_keys = ['spans', 'text', 'target', 'sentiment']
            for key in minimum_keys:
                if not self[key] == other[key]:
                    return False
        return True

class TargetCollection(MutableMapping):
    '''
    Mutable data store for Target data types.  i.e. A data store that contains
    multiple Target instances. This collection ensures that there are no two
    Target instances stored that have the same ID this is because the storage
    of the collection is an OrderedDict.

    Overrides collections.abs.MutableMapping abstract class.

    Functions:

    1. add -- Given a Target instance with an `id` key adds it to the data \
    store.
    '''

    def __init__(self, list_of_target=None):
        '''
        :param list_of_target: An interator of Target instances e.g. a List of \
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
        # instance (value). However if the key does not match the `id` raise
        # KeyError
        if 'id' in value:
            if value['id'] != key:
                raise KeyError('Cannot add this to the data store as the key {}'\
                               ' is not the same as the `id` in the Target '\
                               'instance value {}'.format(key, value))
            del temp_value['id']
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
        out the id of the target if using __setitem__

        :Example:

        >>> target = Target([(10, 16)], '1', 'Iphone',
                            'text with Iphone', 0)
        >>> target_col = TargetCollection()
        # Add method is simpler to use than __setitem__
        >>> target_col.add(target)
        # Example of the __setitem__ method
        >>> target_col[target['id']] = target

        :param value: Target instance with a `id` key
        :type value: Target
        :returns: Nothing. Adds the target instance to the data store.
        :rtype: None
        '''

        if not isinstance(value, Target):
            raise TypeError('All values in this store have to be of type '\
                            'Target not {}'.format(type(value)))
        if 'id' not in value:
            raise ValueError('The Target instance given {} does not have an ID'\
                             .format(value))
        self[value['id']] = value

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

    def sentiment_data(self, mapper=None):
        '''
        :param mapper: A dictionary that maps the keys to the values where the \
        keys are the current unique sentiment values of the target instances \
        stored
        :type mapper: dict
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
            return [mapper[target_data['sentiment']]\
                    for target_data in self.values()]

        return [target_data['sentiment'] for target_data in self.values()]

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
