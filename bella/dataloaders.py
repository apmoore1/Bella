import json

from keras.utils import Sequence
import numpy as np

class TargetSequence(Sequence):

    def __init__(self, json_fp, batch_size, tokeniser, sort_field='text'):
        data = []
        self.label_mapper = {-1: 0, 0: 1, 1: 2}
        self.inv_label_mapper = {0: -1, 1: 0, 2: 1}
        self.batch_size = batch_size
        self.tokeniser = tokeniser
    
        with json_fp.open('r') as json_file:
            for line in json_file:
                data.append(json.loads(line))
        data = sorted(data, key=self.sort_by(sort_field))
        self.texts = []
        self.targets = []
        self.labels = []
        for target in data:
            self.texts.append(target['text'])
            self.targets.append(target['target'])
            self.labels.append(target['sentiment'])
    
    def __len__(self):
        return int(np.ceil(len(self.texts) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_texts = self.texts[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_targets = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return [batch_texts, batch_targets, batch_labels]

    def sort_by(self, field):
        def field_sort(target):
            return len(target[field].split())
        return field_sort