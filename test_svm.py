from train_svm import *

loaded_model = pickle.load(open("svm.pkl", 'rb'))

model = DeepSpeakerModel()
model.m.load_weights('/home/nguyendat/Documents/projects/PetProject/VoiceVerification/deep-speaker/checkpoints-triplets/ResCNN_triplet_training_checkpoint_265.h5', by_name=True)


mfcc1 = sample_from_mfcc(read_mfcc('/home/nguyendat/Documents/projects/PetProject/VoiceVerification/deep-speaker/samples/train/5-F-27/5.wav', SAMPLE_RATE), NUM_FRAMES)
mfcc2 = sample_from_mfcc(read_mfcc('/home/nguyendat/Documents/projects/PetProject/VoiceVerification/deep-speaker/samples/train/5-F-27/5-2.wav', SAMPLE_RATE), NUM_FRAMES)
mfcc3 = sample_from_mfcc(read_mfcc('/home/nguyendat/Documents/projects/PetProject/VoiceVerification/deep-speaker/samples/train/6-M-45/6.wav', SAMPLE_RATE), NUM_FRAMES)

feature1 = model.m.predict(np.expand_dims(mfcc1, axis=0))
feature2 = model.m.predict(np.expand_dims(mfcc2, axis=0))
feature3 = model.m.predict(np.expand_dims(mfcc3, axis=0))

cost = batch_cosine_similarity(feature1, feature2)[0].reshape(1, -1)
cost1 = batch_cosine_similarity(feature1, feature3)[0].reshape(1, -1)

print("CHECK_SAME_SPEAKER: ", loaded_model.predict(cost))
print("CHECK_DIFF_SPEAKER: ", loaded_model.predict(cost1))