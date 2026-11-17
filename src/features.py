"""Track listing, track-level split and log-mel segment extraction for GTZAN."""
import os

import librosa
import numpy as np
from sklearn.model_selection import train_test_split

SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
NUM_SEGMENTS = 10
SAMPLES_PER_SEGMENT = SAMPLES_PER_TRACK // NUM_SEGMENTS  # 3 s
N_FFT = 2048
N_MELS = 128

VALID_EXTENSIONS = ('.wav', '.au', '.mp3', '.m4a')
GTZAN_GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# jazz.00054 is unreadable in the widely used Kaggle .wav copy of GTZAN.
# It is excluded by name so that both copies use the same 999-track list.
KNOWN_CORRUPT = {'jazz.00054'}


def list_genres(data_dir):
    return sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))


def list_audio_files(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTENSIONS) and not f.startswith('._'))


def list_tracks(data_dir, genres):
    paths, labels = [], []
    for label_idx, genre in enumerate(genres):
        genre_folder = os.path.join(data_dir, genre)
        for f in list_audio_files(genre_folder):
            if os.path.splitext(f)[0] in KNOWN_CORRUPT:
                print(f"Excluding known corrupted file: {f}")
                continue
            paths.append(os.path.join(genre_folder, f))
            labels.append(label_idx)
    return np.array(paths), np.array(labels)


def split_tracks(paths, labels, seed=42):
    """Split the TRACK list 70/15/15, stratified by genre, before any segmentation."""
    p_train, p_tmp, l_train, l_tmp = train_test_split(paths, labels, test_size=0.30, stratify=labels, random_state=seed)
    p_val, p_test, l_val, l_test = train_test_split(p_tmp, l_tmp, test_size=0.50, stratify=l_tmp, random_state=seed)

    assert not set(p_train) & set(p_val)
    assert not set(p_train) & set(p_test)
    assert not set(p_val) & set(p_test)
    return (p_train, l_train), (p_val, l_val), (p_test, l_test)


def log_mel(segment, hop_length):
    mel = librosa.feature.melspectrogram(y=segment, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=hop_length, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max)


def signal_to_segments(signal, hop_length, max_segments=None):
    """Cut a signal into full 3 s segments and return their log-mel spectrograms, shape (n, 128, frames, 1)."""
    n = len(signal) // SAMPLES_PER_SEGMENT
    if max_segments is not None:
        n = min(n, max_segments)
    segments = [log_mel(signal[d * SAMPLES_PER_SEGMENT:(d + 1) * SAMPLES_PER_SEGMENT], hop_length) for d in range(n)]
    return np.array(segments)[..., np.newaxis]


def extract_segments(paths, labels, hop_length):
    """Segment every track of one split. Returns X, y and the track index of each segment."""
    X, y, track_ids = [], [], []
    for t, (path, label) in enumerate(zip(paths, labels)):
        signal, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        # Tracks slightly shorter than 30 s lose their last segment
        segments = signal_to_segments(signal, hop_length, max_segments=NUM_SEGMENTS)
        X.append(segments)
        y.extend([label] * len(segments))
        track_ids.extend([t] * len(segments))
    return np.concatenate(X), np.array(y), np.array(track_ids)
