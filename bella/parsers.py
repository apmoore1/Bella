'''
Functions that parse the annotated data that is being used in this project. The
annotated dataset are the following:

1. `Li Dong <http://goo.gl/5Enpu7>`_ which links to :py:func:`bella.parsers.dong`
2. Semeval parser
'''
import json
import os
import re
import xml.etree.ElementTree as ET

import ftfy

from bella.data_types import Target, TargetCollection

def dong(file_path):
    '''
    Given file path to the
    `Li Dong <https://github.com/bluemonk482/tdparse/tree/master/data/lidong>`_
    sentiment data it will parse the data and return it as a list of dictionaries.

    :param file_path: File Path to the annotated data
    :type file_path: String
    :returns: A TargetCollection containing Target instances.
    :rtype: TargetCollection
    '''

    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError('This file does not exist {}'.format(file_path))
    file_name, _ = os.path.splitext(os.path.basename(file_path))
    sentiment_range = [-1, 0, 1]

    sentiment_data = TargetCollection()
    with open(file_path, 'r') as dong_file:
        sent_dict = {}
        for index, line in enumerate(dong_file):
            divisible = index + 1
            line = line.strip()
            if divisible % 3 == 1:
                sent_dict['text'] = line
            elif divisible % 3 == 2:
                sent_dict['target'] = line
            elif divisible % 3 == 0:
                sentiment = int(line)
                if sentiment not in sentiment_range:
                    raise ValueError('The sentiment has to be one of the '\
                                     'following values {} not {}'\
                                     .format(sentiment_range, sentiment))
                sent_dict['sentiment'] = int(line)
                text = sent_dict['text'].lower()
                target = sent_dict['target'].lower()
                offsets = [match.span() for match in re.finditer(target, text)]
                if len(target.split()) > 1:
                    joined_target = ''.join(target.split())
                    offsets.extend([match.span()
                                    for match in re.finditer(joined_target, text)])
                sent_dict['spans'] = offsets
                sent_id = file_name + str(len(sentiment_data))
                # Sentence ID is the same as the target as there is only one
                # target per sentence
                sent_dict['sentence_id'] = sent_id
                sent_dict['target_id'] = sent_id
                sent_target = Target(**sent_dict)
                sentiment_data.add(sent_target)
                sent_dict = {}
            else:
                raise Exception('Problem')
    return sentiment_data




def _semeval_extract_data(sentences, file_name, conflict=False,
                          sentence_ids_skip=None):
    '''
    :param sentences: A `sentences` named element
    :param file_name: Name of the file being parsed
    :param conflict: Determine if to keep the target data that has a conflict \
    sentiment label.
    :param sentence_ids_skip: IDs of sentences that should be skipped
    :type sentences: xml.etree.ElementTree.Element
    :type file_name: String
    :type conflict: bool. Defailt False
    :type sentence_ids_skip: list. Default None
    :returns: A TargetCollection containing Target instances.
    :rtype: TargetCollection
    '''

    # Converts the sentiment tags from Strings to ints
    sentiment_mapper = {'conflict' : -2, 'negative' : -1,
                        'neutral' : 0, 'positive' : 1}

    def extract_aspect_terms(aspect_terms, sentence_id):
        '''
        :param aspect_terms: An aspectTerms element within the xml tree
        :param sentence_id: Id of the sentence that the aspects came from.
        :type aspect_terms: xml.etree.ElementTree.Element
        :type sentence_id: String
        :returns: A list of dictioanries containg id, span, sentiment and \
        target
        :rtype: list
        '''

        aspect_terms_data = []
        for index, aspect_term in enumerate(aspect_terms):
            aspect_term = aspect_term.attrib
            aspect_term_data = {}
            sentiment = sentiment_mapper[aspect_term['polarity']]
            if sentiment == -2 and not conflict:
                continue
            aspect_id = '{}{}'.format(sentence_id, index)
            aspect_term_data['target_id'] = aspect_id
            if 'term' in aspect_term:
                aspect_term_data['target'] = aspect_term['term']
            elif 'target' in aspect_term:
                aspect_term_data['target'] = aspect_term['target']
            else:
                raise KeyError('There is no `target` attribute in the opinions '\
                               'element {}'.format(aspect_term))
            aspect_term_data['sentiment'] = sentiment
            aspect_term_data['spans'] = [(int(aspect_term['from']),
                                          int(aspect_term['to']))]
            aspect_term_data['sentence_id'] = sentence_id
            # If the target is NULL then there is no target
            if aspect_term_data['target'] == 'NULL':
                continue
            aspect_terms_data.append(aspect_term_data)
        return aspect_terms_data

    def add_text(aspect_data, text):
        '''
        :param aspect_data: A list of dicts containing `span`, `target` and \
        `sentiment` keys.
        :param text: The text of the sentence that is associated to all of the \
        aspects in the aspect_data list
        :type aspect_data: list
        :type text: String
        :returns: The list of dicts in the aspect_data parameter but with a \
        `text` key with the value that the text parameter contains
        :rtype: list
        '''

        for data in aspect_data:
            data['text'] = text
        return aspect_data

    all_aspect_term_data = TargetCollection()
    for sentence in sentences:
        aspect_term_data = None
        text_index = None
        sentence_id = file_name + sentence.attrib['id']
        # Allow the parser to skip certain sentences
        if sentence_ids_skip is not None:
            if sentence.attrib['id'] in sentence_ids_skip:
                continue
        for index, data in enumerate(sentence):
            if data.tag == 'sentence':
                raise Exception(sentence.attrib['id'])
            if data.tag == 'text':
                text_index = index
            elif data.tag == 'aspectTerms' or data.tag == 'Opinions':
                aspect_term_data = extract_aspect_terms(data, sentence_id)
        if aspect_term_data is None:
            continue
        if text_index is None:
            raise ValueError('A semeval sentence should always have text '\
                             'semeval file {} sentence id {}'\
                             .format(file_name, sentence.attrib['id']))
        sentence_text = sentence[text_index].text
        aspect_term_data = add_text(aspect_term_data, sentence_text)
        for aspect in aspect_term_data:
            sent_target = Target(**aspect)
            all_aspect_term_data.add(sent_target)
    return all_aspect_term_data


def semeval_15_16(file_path, sep_16_from_15=False):
    '''
    Parser for the SemEval 2015 and 2016 datasets.

    :param file_path: File path to the semeval 2014 data
    :param sep_16_from_15: Ensure that the test sets of semeval 2016 is complete \
    seperate from the semeval test set of 2015
    :type file_path: String
    :type sep_16_from_15: bool. Default False
    :returns: A TargetCollection containing Target instances.
    :rtype: TargetCollection
    '''

    file_path = os.path.abspath(file_path)
    file_name, _ = os.path.splitext(os.path.basename(file_path))

    tree = ET.parse(file_path)
    reviews = tree.getroot()
    all_aspect_term_data = []
    if reviews.tag != 'Reviews':
        raise ValueError('The root of all semeval 15/16 xml files should '\
                         'be reviews and not {}'\
                         .format(reviews.tag))
    for review in reviews:
        review_id = review.attrib['rid']
        for sentences in review:
            if sep_16_from_15:
                ids_to_skip = ["en_SnoozeanAMEatery_480032670:4"]
                review_targets = _semeval_extract_data(sentences, file_name,
                                                       sentence_ids_skip=ids_to_skip)
                all_aspect_term_data.extend(review_targets.data())
            else:
                review_targets = _semeval_extract_data(sentences, file_name).data()
                all_aspect_term_data.extend(review_targets)
    return TargetCollection(all_aspect_term_data)

def semeval_14(file_path, conflict=False):
    '''
    Parser for the SemEval 2014 datasets.

    :param file_path: File path to the semeval 2014 data
    :param conflict: determine if to include the conflict sentiment value
    :type file_path: String
    :type conflict: bool. Default False.
    :returns: A TargetCollection containing Target instances.
    :rtype: TargetCollection
    '''
    file_path = os.path.abspath(file_path)
    file_name, _ = os.path.splitext(os.path.basename(file_path))

    tree = ET.parse(file_path)
    sentences = tree.getroot()
    if sentences.tag != 'sentences':
        raise ValueError('The root of all semeval xml files should '\
                         'be sentences and not {}'\
                         .format(sentences.tag))
    return _semeval_extract_data(sentences, file_name, conflict=conflict)

def election(folder_path, include_dnr=False, include_additional=False):
    '''
    Data can be downloaded from
    `FigShare <https://figshare.com/articles/EACL_2017_-_Multi-target_\
    UK_election_Twitter_sentiment_corpus/4479563/1>`_

    :param folder_path: Path to the folder containing the data after it has \
    been unziped and all folders within it have been unziped.
    :param include_dnr: determine if to include the `doesnotapply` label
    :param include_additional: NOTE: This does not work at the moment. \
    Determine if to parse the additional data.
    :type folder_path: String
    :type include_dnr: bool. Default False
    :type include_additional: bool. Default False
    :returns: A TargetCollection containing Target instances.
    :rtype: TargetCollection
    '''

    sentiment_mapper = {'negative' : -1, 'neutral' : 0, 'positive' : 1}
    folder_path = os.path.abspath(folder_path)
    folder_name, _ = os.path.splitext(os.path.basename(folder_path))

    def get_file_data(folder_dir):
        '''
        :param folder_dir: File path to a folder containing JSON data files \
        where the file names is the datas ID
        :type folder_dir: String
        :returns: A dictionary of IDs as keys and JSON data as values
        :rtype: dict
        '''

        data = {}
        for file_name in os.listdir(folder_dir):
            file_path = os.path.join(folder_dir, file_name)
            tweet_id = file_name.rstrip('.json').lstrip('5')
            with open(file_path, 'r') as file_data:
                data[tweet_id] = json.load(file_data)
        return data

    def parse_tweet(tweet_data, anno_data, tweet_id):

        def get_offsets(entity, tweet_text, target):
            offset_shifts = [0, -1, 1]
            from_offset = entity['offset']
            for offset_shift in offset_shifts:
                from_offset_shift = from_offset + offset_shift
                to_offset = from_offset_shift + len(target)
                offsets = [(from_offset_shift, to_offset)]
                offset_text = tweet_text[from_offset_shift : to_offset].lower()
                if offset_text == target.lower():
                    return offsets
            raise ValueError('Offset {} does not match target text {}. Full '\
                             'text {}\nid {}'\
                             .format(from_offset, target, tweet_text, tweet_id))

        def fuzzy_target_match(tweet_text, target):
            low_target = target.lower()
            target_searches = [low_target, r'[^\w]' + low_target,
                               r'[^\w]' + low_target + r'[^\w]',
                               low_target + r'[^\w]',
                               low_target.replace(' ', ''),
                               low_target.replace(" '", '')]
            for target_search in target_searches:
                target_matches = list(re.finditer(target_search,
                                                  tweet_text.lower()))
                if len(target_matches) == 1:
                    return target_matches
            if tweet_id in set(['81211671026352128', '78689580104290305',
                                '81209490499960832']):
                return None
            if tweet_id == '75270720671973376' and target == 'kippers':
                return None
            if tweet_id == '65855178264686592' and target == 'tax':
                return None
            print(tweet_data)
            print(anno_data)
            raise ValueError('Cannot find the exact additional '\
                             'entity {} within the tweet {}'\
                             .format(target, tweet_text))



        target_instances = []
        tweet_id = str(tweet_id)
        tweet_text = tweet_data['content']
        target_ids = []
        # Parse all of the entities that have been detected automatically
        for entity in tweet_data['entities']:
            data_dict = {}
            target = entity['entity']
            target_ids.append(entity['id'])
            entity_id = str(entity['id'])
            data_dict['spans'] = get_offsets(entity, tweet_text, target)
            data_dict['target'] = entity['entity']
            data_dict['target_id'] = folder_name + tweet_id + '#' + entity_id
            data_dict['sentence_id'] = folder_name + tweet_id
            data_dict['sentiment'] = anno_data['items'][entity_id]
            if data_dict['sentiment'] == 'doesnotapply' and not include_dnr:
                continue
            # Convert from Strings to Integer
            data_dict['sentiment'] = sentiment_mapper[data_dict['sentiment']]
            data_dict['text'] = tweet_text
            target_instances.append(Target(**data_dict))
        # Parse all of the entities that have been selected by the user
        if include_additional:
            additional_data = anno_data['additional_items']
            if isinstance(additional_data, dict):
                for target, sentiment in additional_data.items():
                    target_matches = fuzzy_target_match(tweet_text, target)
                    if target_matches is None:
                        continue
                    target_id = max(target_ids) + 1
                    target_ids.append(target_id)
                    data_dict['spans'] = [target_matches[0].span()]
                    data_dict['target'] = target
                    data_dict['sentiment'] = sentiment
                    data_dict['text'] = tweet_text
                    data_dict['sentence_id'] = tweet_id
                    data_dict['target_id'] = tweet_id + '#' + str(target_id)
                    target_instances.append(Target(**data_dict))

        return target_instances

    def get_data(id_file, tweets_data, annos_data):
        targets = []
        with open(id_file, 'r') as id_data:
            for tweet_id in id_data:
                tweet_id = tweet_id.strip()
                tweet_data = tweets_data[tweet_id]
                anno_data = annos_data[tweet_id]
                targets.extend(parse_tweet(tweet_data, anno_data, tweet_id))
        return TargetCollection(targets)

    tweets_data = get_file_data(os.path.join(folder_path, 'tweets'))
    annotations_data = get_file_data(os.path.join(folder_path, 'annotations'))

    train_ids_file = os.path.join(folder_path, 'train_id.txt')
    train_data = get_data(train_ids_file, tweets_data, annotations_data)
    test_ids_file = os.path.join(folder_path, 'test_id.txt')
    test_data = get_data(test_ids_file, tweets_data, annotations_data)

    return train_data, test_data


def hu_liu(file_path):
    '''
    Parser for the datasets from the following two papers (DOES NOT WORK):

    1. `A Holistic Lexicon-Based Approach to Opinion Mining \
    <https://www.cs.uic.edu/~liub/FBS/opinion-mining-final-WSDM.pdf>`_
    2. `Mining and Summarizing Customer Reviews \
    <https://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf>`_

    Currently this does not work. This is due to the dataset not containing
    enough data to determine where the targets are in the text.

    :param file_path: The path to a file containing annotations in the format \
    of hu and liu sentiment datasets.
    :type file_path: String
    :returns: A TargetCollection containing Target instances.
    :rtype: TargetCollection
    '''
    file_path = os.path.abspath(file_path)
    file_name = os.path.basename(file_path)
    sentiment_data = TargetCollection()

    with open(file_path, 'r', encoding='cp1252') as annotations:
        for sentence_index, annotation in enumerate(annotations):
            # If it does not contain ## then not a sentence
            if '##' not in annotation:
                continue
            targets_text = annotation.split('##')
            if len(targets_text) > 2 or len(targets_text) < 1:
                raise ValueError('The annotation {} when split on `##` should '\
                                 'contain at least the sentence text and at'\
                                 ' most the text and the targets and not {}'\
                                 .format(annotation, targets_text))
            # If it just contains the sentence text then go to next
            elif len(targets_text) == 1:
                continue
            elif targets_text[0].strip() == '':
                continue
            targets, text = targets_text
            targets = targets.strip()
            text = text.strip()
            sentence_id = file_name + '#{}'.format(sentence_index)

            targets = targets.split(',')
            for target_index, target in enumerate(targets):
                target = target.strip()
                sentiment_match = re.search(r'\[[+-]\d\]$', target)
                is_implicit = re.search(r'\[[up]\]', target)
                if is_implicit:
                    print('Target {} is implicit {}'.format(target, text))
                    continue
                if not sentiment_match:
                    raise ValueError('Target {} does not have a corresponding'\
                                     ' sentiment value. annotation {}'\
                                     .format(target, annotation))
                target_text = target[:sentiment_match.start()].strip()
                sentiment_text = sentiment_match.group().strip().strip('[]')
                sentiment_value = int(sentiment_text)

                target_matches = list(re.finditer(target_text, text))
                if len(target_matches) != 1:
                    print('The Target {} can only occur once in the '\
                          'text {}'.format(target_text, text))
                    continue
                    raise ValueError('The Target {} can only occur once in the '\
                                     'text {}'.format(target_text, text))
                target_span = target_matches[0].span()
                target_id = sentence_id + '#{}'.format(target_index)

                data_dict = {}
                data_dict['spans'] = [target_span]
                data_dict['target'] = target_text
                data_dict['sentiment'] = sentiment_value
                data_dict['text'] = text
                data_dict['sentence_id'] = sentence_id
                data_dict['target_id'] = target_id
                sentiment_data.add(Target(**data_dict))
    return sentiment_data


def mitchel(file_name):
    '''
    Parser for the dataset introduced by `Mitchel et al. \
    <https://www.aclweb.org/anthology/D13-1171>`_. The dataset can be downloaded
    from `<here http://www.m-mitchell.com/code/MitchellEtAl-13-OpenSentiment.tgz>`_
    the dataset can be found within the tarball under /en/10-fold and then
    choose one of the folds e.g. train_1 and test_1 to get the full dataset.

    :param file_path: path to either the train or test data.
    :type file_path: String
    :returns: A TargetCollection containing Target instances.
    :rtype: TargetCollection
    '''

    def extract_targets(current_target, end_span, start_span, targets,
                        target_spans, target_index, tweet_text, sentiment_data,
                        tweet_id, target_sentiments):
        if current_target != []:
            target_word = ' '.join(current_target)
            end_span = start_span + len(target_word)
            targets.append(target_word)
            target_spans.append((start_span, end_span))
            start_span, end_span = None, None
            current_target = []
            target_index += 1
        tweet_text = ' '.join(tweet_text)
        for index, target in enumerate(targets):
            target_id = '{}#{}'.format(tweet_id, index)
            target_sentiment = target_sentiments[index]
            target_span = target_spans[index]
            if tweet_text[target_span[0] : target_span[1]] != target:
                raise Exception('The target span {} does not match the '\
                                'target word {} in {}'\
                                .format(target_span, target, tweet_text))
            target_data = {'spans' : [target_span], 'target_id' : target_id,
                           'target' : target, 'text' : tweet_text,
                           'sentiment' : target_sentiment,
                           'sentence_id' : tweet_id}
            target_data = Target(**target_data)
            sentiment_data.add(target_data)

        return sentiment_data

    sentiment_mapper = {'negative' : -1, 'neutral' : 0, 'positive' : 1}
    sentiment_data = TargetCollection()

    with open(file_name, 'r') as fp:
        tweet_id = None
        tweet_text = []
        targets = []
        current_target = []
        target_sentiments = []
        target_spans = []
        start_span = None
        end_span = None
        target_index = 0

        for line in fp:
            line = line.strip()
            tweet_id_line = re.match(r'## Tweet (\d+)', line)
            if tweet_id_line is not None:

                if tweet_text != []:
                    sentiment_data = extract_targets(current_target, end_span,
                                                     start_span, targets,
                                                     target_spans,
                                                     target_index, tweet_text,
                                                     sentiment_data, tweet_id,
                                                     target_sentiments)
                    tweet_text = []
                    targets = []
                    current_target = []
                    target_sentiments = []
                    target_spans = []
                    target_index = 0
                    start_span = None
                    end_span = None
                    target_index = 0
                tweet_id = tweet_id_line.group(1)
                continue

            if line == '':
                continue

            line_data = line.split('\t')
            if len(line_data) != 3:
                if len(line_data) == 4:
                    if line_data[2] == 'NUMBER':
                        line_data = line_data[0], line_data[1], line_data[3]
                    else:
                        raise Exception('Cannot parse line {} in Tweet ID {}'\
                                        .format(line, tweet_id))
                else:
                    raise Exception('Cannot parse line {} in Tweet ID {}'.format(line, tweet_id))
            word, ner_data, sentiment = line_data
            if len(word.split()) != 1:
                raise Exception('Why is the word got a space in it: {}'.format(word))
            word = ftfy.fix_encoding(word)
            tweet_text.append(word)

            # Contains sentiment
            if sentiment != '_':
                if len(current_target) != 0:
                    if ner_data[0] == 'B':
                        target_word = ' '.join(current_target)
                        end_span = start_span + len(target_word)
                        targets.append(target_word)
                        target_spans.append((start_span, end_span))
                        start_span, end_span = None, None
                        current_target = []
                        target_index += 1
                    else:
                        raise Exception('Contains the following target {} id {}'\
                                        .format(current_target, tweet_id))
                current_target.append(word)
                sentiment = sentiment_mapper[sentiment]
                target_sentiments.append(sentiment)
                start_span = len(' '.join(tweet_text)) - len(word)
            elif len(current_target) != 0:
                if ner_data[0] == 'I':
                    current_target.append(word)
                else:
                    target_word = ' '.join(current_target)
                    end_span = start_span + len(target_word)
                    targets.append(target_word)
                    target_spans.append((start_span, end_span))
                    start_span, end_span = None, None
                    current_target = []
                    target_index += 1
        sentiment_data = extract_targets(current_target, end_span, start_span,
                                         targets, target_spans, target_index,
                                         tweet_text, sentiment_data, tweet_id,
                                         target_sentiments)
    return sentiment_data
