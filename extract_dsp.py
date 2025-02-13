""" This script extracts various classic DSP features using librosa."""

import sys
import os
import glob
import time
import argparse
from multiprocessing import Pool
import h5py
import numpy as np

import librosa
import essentia.standard as es


def process_audio(
    audio_path: str,
    feat_dir: str,
    log_dir: str,
    sample_rate: float,
    hop_size: int
):

    # Get the YouTube ID of the audio file
    yt_id = os.path.basename(audio_path).split(".")[0]

    try:
        # Load the audio, convert to mono and adjust the sample rate
        audio = es.MonoLoader(filename=audio_path, sampleRate=sample_rate)()

        if len(audio) == 0:
            raise ValueError("Empty audio file.")

        # Estimate a static tempo
        onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_size)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sample_rate)
       
        # We store each file as cqt_dir/yt_id[:2]/yt_id.mm
        output_dir = os.path.join(feat_dir, yt_id[:2])
        os.makedirs(output_dir, exist_ok=True)
        # Save the CQT as memmap
        output_path = os.path.join(output_dir, f"{yt_id}.h5")
        
        with h5py.File(output_path, "a") as f:
            f.create_dataset("tempo", data=tempo)
            f.create_dataset("onset_env", data=onset_env)
            
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing {audio_path}:\n{repr(e)}")
        with open(os.path.join(log_dir, f"{yt_id}.txt"), "w") as out_f:
            out_f.write(repr(e) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "audio_dir",
        type=str,
        help="Directory containing the audio files or a text file containing the audio paths.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Root directory to save the features. <output_dir>/cqt/ will be created.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate to use for the audio files",
    )
    parser.add_argument(
        "--hop-size",
        type=int,
        default=512,
        help="Hop size to use for the CQT in samples.",
    )

    parser.add_argument(
        "--processes",
        type=int,
        default=20,
        help="Number of parallel processes to use for feature extraction.",
    )
    args = parser.parse_args()

    # Load the audio paths
    print("Loading audio files...")
    if os.path.isfile(args.audio_dir):
        with open(args.audio_dir, "r") as f:
            audio_paths = sorted([p.strip() for p in f.readlines()])
    elif os.path.isdir(args.audio_dir):
        audio_paths = sorted(
            glob.glob(os.path.join(args.audio_dir, "**", "*.mp4"), recursive=True)
        )
    else:
        raise ValueError("audio_dir must be a directory or a file.")
    print(f"{len(audio_paths):,} audio files found.")

    # Skip previously computed features in output_dir
    print("Checking for previously computed features...")
    old_audio_paths = glob.glob(
        os.path.join(args.output_dir, "**", "*.h5"), recursive=True
    )
    old_audio_ids = set([os.path.basename(p).split(".")[0] for p in old_audio_paths])
    audio_paths = [
        p for p in audio_paths if os.path.basename(p).split(".")[0] not in old_audio_ids
    ]
    print(f"{len(audio_paths):,} new features will be computed.")
    del old_audio_paths, old_audio_ids

    # Create the output directories
    feat_dir = os.path.join(args.output_dir, "h5")
    os.makedirs(feat_dir, exist_ok=True)
    print(f"Features will be saved in {feat_dir}")
    log_dir = os.path.join(args.output_dir, "feat_logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logs will be saved in {log_dir}")

    # Extract the CQTs
    t0 = time.monotonic()
    print(f"Extracting features with {args.processes} processes...")
    with Pool(processes=args.processes) as pool:
        pool.starmap(
            process_audio,
            [
                (
                    audio_path,
                    feat_dir,
                    log_dir,
                    args.sample_rate,
                    args.hop_size
                )
                for audio_path in audio_paths
            ],
        )
    print(f"Extraction took {time.monotonic()-t0:.2f} seconds.")
