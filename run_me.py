import os
import argparse

from audio import Audio
from batcher import KerasFormatConverter
from constants import SAMPLE_RATE, NUM_FRAMES
from test import test
from train import start_training
from utils import ensures_dir

def main(args):
    ensures_dir(args.working_dir)

    if args.preprocess:
        if args.audio_dir is None:
            return
        Audio(cache_dir=args.working_dir, audio_dir=args.audio_dir, sample_rate=args.sample_rate)

    if args.build_keras_inputs:
        counts_per_speaker = [int(b) for b in args.counts_per_speaker.split(',')]
        kc = KerasFormatConverter(args.working_dir)
        kc.generate(max_length=NUM_FRAMES, counts_per_speaker=counts_per_speaker)
        kc.persist_to_disk()

    if args.train_embedding:
        if args.pre_training_phase:
            start_training(args.working_dir, pre_training_phase=args.pre_training_phase)
        start_training(args.working_dir, False)

    if args.train_classifier:
        start_training(args.working_dir, pre_training_phase=False, classify=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type=str, default='./')
    parser.add_argument('--audio_dir', type=str, default='samples/train')
    parser.add_argument('--sample_rate', default=SAMPLE_RATE, type=int)
    parser.add_argument('--counts_per_speaker', default='600,100', type=str) # So luong train, test

    parser.add_argument('--preprocess', default=1, type=int)
    parser.add_argument('--build_keras_inputs', default=1, type=int)
    parser.add_argument('--train_embedding', default=1, type=int)
    parser.add_argument('--pre_training_phase', default=1, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--train_classifier', default=1, type=int)

    args = parser.parse_args()
    main(args)
