## Deep Speaker: An End-to-End Neural Speaker Embedding System.
Unofficial Keras implementation of Deep Speaker | [Paper](https://arxiv.org/pdf/1705.02304.pdf) | [Pretrained Models](https://drive.google.com/open?id=18h2bmsAWrqoUMsh_FQHDDxp7ioGpcNBa)

### Sample Results


 *Model name* | *Training dataset* | *Num speakers* | *ACC* | *EER* | Download model
 | :--- | :--- | :--- | :--- | :--- | :--- |
ResCNN Softmax trained          | LibriSpeech all | 2484 | 0.997 | 0.043 | [Click](https://drive.google.com/open?id=1wCeoa99XfO5r0OX1K5VjJUaCbpvEx9cc)
ResCNN Softmax+Triplet trained  | LibriSpeech all | 2484 | 0.996 | 0.025 | [Click](https://drive.google.com/open?id=1SJBmHpnaW1VcbFWP6JfvbT3wWP9PsqxS)


### Overview

Deep Speaker is a neural speaker embedding system that maps utterances to a hypersphere where speaker similarity is measured by cosine similarity. The embeddings generated by Deep Speaker can be used for many tasks, including speaker identification,
verification, and clustering.

## Getting started
### Install dependencies
#### Requirements
- tensorflow>=2.0
- keras>=2.3.1
```bash
pip install -r requirements.txt
```

### Training

The code for training is available in this repository. 

System requirements for a complete training are:
- At least 100GB of free disk space.
- 32GB of memory.
- A NVIDIA GPU such as the 1080Ti.

```
./deep-speaker download_librispeech
./deep-speaker build_mfcc
./deep-speaker build_model_inputs
./deep-speaker train_softmax           # takes ~3 days.
./deep-speaker train_triplet           # takes ~3 days.
```

### Test instruction using pretrained model
- Download the trained models
 

 *Model name* | *Used datasets for training* | *Num speakers* | *Model Link* | 
 | :--- | :--- | :--- | :--- |
ResCNN Softmax trained  | LibriSpeech train-clean-360 | 921 | [Click](https://drive.google.com/open?id=1wCeoa99XfO5r0OX1K5VjJUaCbpvEx9cc)
ResCNN Softmax+Triplet trained  | LibriSpeech all | 2484 | [Click](https://drive.google.com/open?id=1SJBmHpnaW1VcbFWP6JfvbT3wWP9PsqxS)

* Run with pretrained model

```python
import numpy as np

from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES
from conv_models import DeepSpeakerModel
from test import batch_cosine_similarity

# Reproducible results.
np.random.seed(123)

# Define the model here.
model = DeepSpeakerModel()

# Load the checkpoint.
model.m.load_weights('/Users/premy/deep-speaker/checkpoints/ResCNN_triplet_training_checkpoint_175.h5', by_name=True)

# Sample some inputs for WAV/FLAC files for the same speaker.
mfcc_001 = sample_from_mfcc(read_mfcc('samples/PhilippeRemy/PhilippeRemy_001.wav', SAMPLE_RATE), NUM_FRAMES)
mfcc_002 = sample_from_mfcc(read_mfcc('samples/PhilippeRemy/PhilippeRemy_002.wav', SAMPLE_RATE), NUM_FRAMES)

# Call the model to get the embeddings of shape (1, 512) for each file.
predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
predict_002 = model.m.predict(np.expand_dims(mfcc_002, axis=0))

# Do it again with a different speaker.
mfcc_003 = sample_from_mfcc(read_mfcc('samples/1255-90413-0001.flac', SAMPLE_RATE), NUM_FRAMES)
predict_003 = model.m.predict(np.expand_dims(mfcc_003, axis=0))

# Compute the cosine similarity and check that it is higher for the same speaker.
print('SAME SPEAKER', batch_cosine_similarity(predict_001, predict_002)) # SAME SPEAKER [0.81564593]
print('DIFF SPEAKER', batch_cosine_similarity(predict_001, predict_003)) # DIFF SPEAKER [0.1419204]
```

* Commands to reproduce the test results after the training


```bash
$ export CUDA_VISIBLE_DEVICES=0; python cli.py test-model --working_dir /home/philippe/ds-test/triplet-training/ --
checkpoint_file ../ds-test/checkpoints-softmax/ResCNN_checkpoint_102.h5
f-measure = 0.789, true positive rate = 0.733, accuracy = 0.996, equal error rate = 0.043
```

```bash
$ export CUDA_VISIBLE_DEVICES=0; python cli.py test-model --working_dir /home/philippe/ds-test/triplet-training/ --checkpoint_file ../ds-test/checkpoints-triplets/ResCNN_checkpoint_175.h5
f-measure = 0.849, true positive rate = 0.798, accuracy = 0.997, equal error rate = 0.025
```
