from pathlib import Path
import sys
a_path = str(Path('..').resolve())
sys.path.insert(0, a_path)

from tensorflow.keras.utils import Sequence

from bella.parsers import semeval_14
from bella.preprocessing import tokeniser
from bella.tokenisers import moses, stanford
from bella.word_embeddings import GloveCommonEmbedding
from bella.dataloaders import LeftRightTargetSequence, TargetSequence
from bella.keras_models import tclstm

rest_sem_dir = Path('..', '..', 'aspect datasets', 'semeval_2014')
rest_train = semeval_14(Path(rest_sem_dir, 'restaurants_train.xml'))
rest_test = semeval_14(Path(rest_sem_dir, 'restaurants_test.xml'))

rest_train_fp, rest_dev_fp = rest_train.to_json_file(['restaurants train', 'restaurants dev'], 
                                                     0.2, cache=False, random_state=42)
rest_test_fp = rest_test.to_json_file('restaurants test', cache=False)

tok = tokeniser(rest_train_fp, rest_dev_fp, rest_test_fp, 
                tokeniser_function=moses, 
                lower=True, filters='', oov_token='<UNK>')
train_data = LeftRightTargetSequence(rest_train_fp, 32, tok, stanford)
val_data = LeftRightTargetSequence(rest_dev_fp, 32, tok, stanford)



import numpy as np


glove_rest = GloveCommonEmbedding(840, tok.word_index).embedding

model = tclstm(glove_rest, lstm_size=300, target_embedding=glove_rest, dropout=0.5)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

model.fit_generator(train_data, validation_data=val_data, shuffle=True, 
                    use_multiprocessing=True, workers=4, epochs=10)