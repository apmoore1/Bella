'''
A list of functions that convert TargetCollections into a specified file format.

1. semeval_14: Converts the TargetCollection into XML file format that was used \
in `SemEval 2014 task 4 <http://alt.qcri.org/semeval2014/task4/>`_
'''
import xml.etree.ElementTree as ET

def semeval_14(file_path, target_collection):
    '''
    :param file_path: Location of the file to save the XML data to
    :param target_collection: The target collection data to convert into the \
    XML file.
    :type file_path: String
    :type target_collection: TargetCollection
    :returns: Nothing. Saves the data to the file path given.
    :rtype: None
    '''
    sentiment_mapper = {1 : 'positive', 0 : 'neutral', -1 : 'negative'}
    sentence_targets = target_collection.group_by_sentence()
    tree = ET.Element('sentences')
    for sentence_id, targets in sentence_targets.items():
        # Removes part of the sentence id we add from our parsing
        if sentence_id.startswith('samsung_galaxy_s5'):
            sentence_id = sentence_id[len('samsung_galaxy_s5'):]
        sentence_element = ET.SubElement(tree, 'sentence',
                                         attrib={'id' : sentence_id})
        text_element = ET.SubElement(sentence_element, 'text')
        text_element.text = targets[0]['text']
        aspects_element = ET.SubElement(sentence_element, 'aspectTerms')
        for target in targets:
            spans = target['spans']
            if len(spans) > 1:
                raise ValueError('There should only be one set of spans not'\
                                 '{}. List of spans: {}'\
                                 .format(len(spans), spans))
            span_from, span_to = spans[0]
            attributes = {'term' : target['target'],
                          'polarity' : sentiment_mapper[target['sentiment']],
                          'from' : str(span_from), 'to' : str(span_to)}
            aspect_element = ET.SubElement(aspects_element, 'aspectTerm',
                                           attrib=attributes)

    with open(file_path, 'w') as xml_file:
        tree = ET.ElementTree(tree)
        tree.write(xml_file, encoding='unicode', xml_declaration=True)
