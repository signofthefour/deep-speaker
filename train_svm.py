import numpy as np
import random
from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES
from conv_models import DeepSpeakerModel
from test import batch_cosine_similarity
from sklearn import svm
import os
import pickle

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
        # print(utt1_path)
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
        files=os.listdir(speaker1_path)
        utt1_dir = random.choice(files)
        files=os.listdir(speaker2_path)
        utt2_dir = random.choice(files)
        utt1_path = os.path.join(speaker1_path, utt1_dir)
        utt2_path = os.path.join(speaker2_path, utt2_dir)
        # print(utt1_path)
        mfcc1 = sample_from_mfcc(read_mfcc(utt1_path, SAMPLE_RATE), NUM_FRAMES)
        mfcc2 = sample_from_mfcc(read_mfcc(utt2_path, SAMPLE_RATE), NUM_FRAMES)
        return mfcc1, mfcc2, label

def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.multiply(x1, x2)
    # s = np.sum(mul, axis=1)

    # l1 = np.sum(np.multiply(x1, x1),axis=1)
    # l2 = np.sum(np.multiply(x2, x2), axis=1)
    # as values have have length 1, we don't need to divide by norm (as it is 1)
    return mul

# np.random.seed(123)
# random.seed(123)

def main():

    model = DeepSpeakerModel()
    model.m.load_weights('/home/nguyendat/Documents/projects/PetProject/VoiceVerification/deep-speaker/checkpoints-triplets/ResCNN_triplet_training_checkpoint_265.h5', by_name=True)

    # mfcc_001 = sample_from_mfcc(read_mfcc('/home/nguyendat/Documents/projects/PetProject/VoiceVerification/deep-speaker/samples/train/5-F-27/5.wav', SAMPLE_RATE), NUM_FRAMES)
    # mfcc_002 = sample_from_mfcc(read_mfcc('/home/nguyendat/Documents/projects/PetProject/VoiceVerification/deep-speaker/samples/train/5-F-27/5-2.wav', SAMPLE_RATE), NUM_FRAMES)

    # predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
    # predict_002 = model.m.predict(np.expand_dims(mfcc_002, axis=0))

    # mfcc_003 = sample_from_mfcc(read_mfcc('/home/nguyendat/Documents/projects/PetProject/VoiceVerification/deep-speaker/samples/train/6-M-45/6.wav', SAMPLE_RATE), NUM_FRAMES)
    # predict_003 = model.m.predict(np.expand_dims(mfcc_003, axis=0))

    # print('SAME SPEAKER', batch_cosine_similarity(predict_001, predict_002))
    # print('DIFF SPEAKER', batch_cosine_similarity(predict_001, predict_003))
    features = []
    labels = []
    for x in range(10):
        mfcc1, mfcc2, label = load_data()
        feature1 = model.m.predict(np.expand_dims(mfcc1, axis=0))
        feature2 = model.m.predict(np.expand_dims(mfcc2, axis=0))
        cost = batch_cosine_similarity(feature1, feature2)
        # print(cost)
        features.append(cost[0])
        labels.append(label)
    # print(cost.shape)
    #  load 2 file (random) + label, predict roi dua vao SVM,
    # dung den triplet
    # features = feature1 + feature2
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    features = np.array(features)
    labels = np.array(labels)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(features,labels) 
    svm_pickle = open('svm.pkl','wb')
    pickle.dump(clf,svm_pickle)
    svm_pickle.close()

    # load_data()

if __name__ == "__main__":
    main()