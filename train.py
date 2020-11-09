import logging
import os

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tqdm import tqdm

from batcher import KerasFormatConverter, LazyTripletBatcher
from constants import BATCH_SIZE, CHECKPOINTS_SOFTMAX_DIR, CHECKPOINTS_TRIPLET_DIR, CHECKPOINTS_CLASSIFY_DIR, NUM_FRAMES, NUM_FBANKS, SAMPLE_RATE
from conv_models import DeepSpeakerModel
from triplet_loss import deep_speaker_loss
from utils import load_best_checkpoint, ensures_dir
from sklearn import svm
from audio import read_mfcc
from batcher import sample_from_mfcc
import numpy as np
import pickle

logger = logging.getLogger(__name__)

# Otherwise it's just too much logging from Tensorflow...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_data(train_dir='./samples/train'):
    mfccs=[]
    y=[]
    import random
    label = random.randint(0,1)
    subfolders= os.listdir(train_dir)
    if label:
        speaker_dir = random.choice(subfolders)
        speaker_path = os.path.join(train_dir, speaker_dir)
        files=os.listdir(speaker_path)
        utt1_dir = random.choice(files)
        utt2_dir = random.choice(files)
        while utt2_dir == utt1_dir and len(files) > 1:
            utt2_dir = random.choice(files)
        utt1_path = os.path.join(speaker_path, utt1_dir)
        utt2_path = os.path.join(speaker_path, utt2_dir)
        mfcc1 = sample_from_mfcc(read_mfcc(utt1_path, SAMPLE_RATE), NUM_FRAMES)
        mfcc2 = sample_from_mfcc(read_mfcc(utt2_path, SAMPLE_RATE), NUM_FRAMES)
        return mfcc1, mfcc2, label
    
    else:
        speaker1_dir = random.choice(subfolders)
        speaker2_dir = random.choice(subfolders)
        while speaker1_dir == speaker2_dir:
            speaker2_dir = random.choice(subfolders)
        speaker1_path = os.path.join(train_dir, speaker1_dir)
        speaker2_path = os.path.join(train_dir, speaker2_dir)
        files=os.listdir(speaker1_dir)
        utt1_dir = random.choice(files)
        files=os.listdir(speaker2_dir)
        utt2_dir = random.choice(files)
        utt1_path = os.path.join(speaker1_path, utt1_dir)
        utt2_path = os.path.join(speaker1_path, utt2_dir)
        mfcc1 = sample_from_mfcc(read_mfcc(utt1_path, SAMPLE_RATE), NUM_FRAMES)
        mfcc2 = sample_from_mfcc(read_mfcc(utt2_path, SAMPLE_RATE), NUM_FRAMES)
        return mfcc1, mfcc2, label
    # for label in labels:
    #     labelPath = os.path.join(train_dir,label)
    #     subfolders=os.listdir(labelPath)
    #     for subfolder in subfolders:
    #         subfolderPath = os.path.join(labelPath,subfolder)
    #         files=os.listdir(subfolderPath)
    #         for file in files:
    #             mfcc = sample_from_mfcc(read_mfcc(os.path.join(subfolderPath,file), SAMPLE_RATE), NUM_FRAMES)
    #             mfccs.append(mfcc)
    #             y.append(label)
    # return np.array(mfccs),np.array(y)

def fit_model(dsm: DeepSpeakerModel, working_dir: str, max_length: int = NUM_FRAMES, batch_size=BATCH_SIZE, epochs=1000, classify=False, initial_epoch=0):
    batcher = LazyTripletBatcher(working_dir, max_length, dsm, classify=classify)

    # build small test set.
    test_batches = []
    for _ in tqdm(range(200), desc='Build test set'):
        test_batches.append(batcher.get_batch_test(batch_size, classify=classify))

    def test_generator():
        while True:
            for bb in test_batches:
                yield bb

    def train_generator():
        while True:
            yield batcher.get_random_batch(batch_size, is_test=False, classify=classify)

    checkpoint_name = dsm.m.name + '_checkpoint'

    if classify:
        checkpoint_filename = os.path.join(CHECKPOINTS_CLASSIFY_DIR, checkpoint_name + '_{epoch}.h5')
    else:
        checkpoint_filename = os.path.join(CHECKPOINTS_TRIPLET_DIR, checkpoint_name + '_{epoch}.h5')
    

    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=checkpoint_filename, save_best_only=True)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.001, patience=20, verbose=1)

    dsm.m.fit_generator(train_generator(), steps_per_epoch=2000, shuffle=False,
              epochs=epochs, validation_data=test_generator(), validation_steps=len(test_batches),
              initial_epoch=initial_epoch, callbacks=[checkpoint, early_stopping])

    mfcc1, mfcc2, label = load_data(os.path.join(working_dir,'samples/train'))
    feature1 = dsm.m.predict(np.expand_dims(mfcc1, axis=0))
    feature2 = dsm.m.predict(np.expand_dims(mfcc2, axis=0))
    #  load 2 file (random) + label, predict roi dua vao SVM,
    # dung den triplet
    features = [feature1, feature2]
    clf = svm.SVC()
    clf.fit(features,label) 
    svm_pickle = open('svm.pkl','wb')
    pickle.dump(clf,svm_pickle)
    svm_pickle.close()

def fit_model_softmax(dsm: DeepSpeakerModel, kx_train, ky_train, kx_test, ky_test,
                      batch_size=BATCH_SIZE, max_epochs=1000, initial_epoch=0):
    checkpoint_name = dsm.m.name + '_checkpoint'
    checkpoint_filename = os.path.join(CHECKPOINTS_SOFTMAX_DIR, checkpoint_name + '_{epoch}.h5')
    checkpoint = ModelCheckpoint(monitor='val_accuracy', filepath=checkpoint_filename, save_best_only=True)

    # if the accuracy does not increase by 0.1% over 20 epochs, we stop the training.
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=20, verbose=1, mode='max')

    # if the accuracy does not increase over 10 epochs, we reduce the learning rate by half.
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, min_lr=0.0001, verbose=1)

    max_len_train = len(kx_train) - len(kx_train) % batch_size
    kx_train = kx_train[0:max_len_train]
    ky_train = ky_train[0:max_len_train]
    max_len_test = len(kx_test) - len(kx_test) % batch_size
    kx_test = kx_test[0:max_len_test]
    ky_test = ky_test[0:max_len_test]

    dsm.m.fit(x=kx_train,
              y=ky_train,
              batch_size=batch_size,
              epochs=initial_epoch + max_epochs,
              initial_epoch=initial_epoch,
              verbose=1,
              shuffle=True,
              validation_data=(kx_test, ky_test),
              callbacks=[early_stopping, reduce_lr, checkpoint])


def start_training(working_dir, pre_training_phase=True, epochs=1000, classify=False):
    ensures_dir(CHECKPOINTS_SOFTMAX_DIR)
    ensures_dir(CHECKPOINTS_TRIPLET_DIR)
    ensures_dir(CHECKPOINTS_CLASSIFY_DIR)
    batch_input_shape = [None, NUM_FRAMES, NUM_FBANKS, 1]
    if pre_training_phase:
        logger.info('Softmax pre-training.')
        kc = KerasFormatConverter(working_dir)
        num_speakers_softmax = len(kc.categorical_speakers.speaker_ids)
        dsm = DeepSpeakerModel(batch_input_shape, include_softmax=True, num_speakers_softmax=num_speakers_softmax)
        dsm.m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        pre_training_checkpoint = load_best_checkpoint(CHECKPOINTS_SOFTMAX_DIR)
        if pre_training_checkpoint is not None:
            initial_epoch = int(pre_training_checkpoint.split('/')[-1].split('.')[0].split('_')[-1])
            logger.info(f'Initial epoch is {initial_epoch}.')
            logger.info(f'Loading softmax checkpoint: {pre_training_checkpoint}.')
            dsm.m.load_weights(pre_training_checkpoint)  # latest one.
        else:
            initial_epoch = 0
        fit_model_softmax(dsm, kc.kx_train, kc.ky_train, kc.kx_test, kc.ky_test, initial_epoch=initial_epoch, max_epochs=epochs)
    else:
        logger.info('Training with the triplet loss.')
        kc = KerasFormatConverter(working_dir)
        num_speakers_softmax = len(kc.categorical_speakers.speaker_ids)

        dsm = DeepSpeakerModel(batch_input_shape, include_softmax=False, include_classifier=False, num_speakers_softmax=num_speakers_softmax)
        classify_checkpoint = load_best_checkpoint(CHECKPOINTS_CLASSIFY_DIR)
        triplet_checkpoint = load_best_checkpoint(CHECKPOINTS_TRIPLET_DIR)
        pre_training_checkpoint = load_best_checkpoint(CHECKPOINTS_SOFTMAX_DIR)
        initial_epoch = 0

        if classify:
            if triplet_checkpoint:
                logger.info(f'Loading triplet checkpoint: {triplet_checkpoint}.')
                dsm.m.load_weights(triplet_checkpoint, by_name=True)
            if classify_checkpoint:
                dsm.m.load_weights(classify_checkpoint)
                initial_epoch = int(classify_checkpoint.split('/')[-1].split('.')[0].split('_')[-1])
            elif pre_training_checkpoint:
                logger.info(f'Loading pre-training checkpoint: {pre_training_checkpoint}.')
                dsm.m.load_weights(pre_training_checkpoint, by_name=True)

        else:
            if triplet_checkpoint is not None:
                logger.info(f'Loading triplet checkpoint: {triplet_checkpoint}.')
                dsm.m.load_weights(triplet_checkpoint)
                initial_epoch = int(triplet_checkpoint.split('/')[-1].split('.')[0].split('_')[-1])

            elif pre_training_checkpoint is not None:
                logger.info(f'Loading pre-training checkpoint: {pre_training_checkpoint}.')
                # If `by_name` is True, weights are loaded into layers only if they share the
                # same name. This is useful for fine-tuning or transfer-learning models where
                # some of the layers have changed.
                dsm.m.load_weights(pre_training_checkpoint, by_name=True)

        if not classify:
            dsm.m.compile(optimizer=SGD(), loss=deep_speaker_loss)
        else:
            dsm.m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        fit_model(dsm, working_dir, NUM_FRAMES, epochs=epochs, classify=classify, initial_epoch=initial_epoch)
        print('Finished========================')