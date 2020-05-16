import random

import numpy as np

from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES
from conv_models import DeepSpeakerModel
from test import batch_cosine_similarity
from batcher import KerasFormatConverter

kc = KerasFormatConverter('./')
# Define the model here.
model = DeepSpeakerModel(include_softmax=False, include_classifier=True, num_speakers_softmax=len(kc.categorical_speakers.speaker_ids))

# Load the checkpoint.
model.m.load_weights('checkpoints-classify/ResCNN_checkpoint_1.h5')

mfcc_001 = sample_from_mfcc(read_mfcc('samples/train/0/0/0-0-Recording (12).m4a', SAMPLE_RATE), NUM_FRAMES)
predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
print(np.argmax(predict_001[0]))
