"""Tune, train and evaluate the CNN on GTZAN (same pipeline as the notebook).

    python src/train.py --data-dir data/genres                       # search + final model + test
    python src/train.py --data-dir data/genres --config outputs/best_config.json   # skip the search
"""
import argparse
import json
import os
import random

import numpy as np
import tensorflow as tf

from features import extract_segments, list_genres, list_tracks, split_tracks
from model import build_cnn, build_tunable_model

SEED = 42
HOP_LENGTH_OPTIONS = [512, 256]


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def tune(train, val, num_classes, output_dir, max_trials, epochs):
    """Random search per hop length; selection uses validation accuracy only."""
    import keras_tuner as kt

    best_val, best_config, val_acc_per_hop = 0.0, None, {}
    for hop in HOP_LENGTH_OPTIONS:
        print(f"\nExtracting features with hop length {hop}")
        X_train, y_train, _ = extract_segments(*train, hop)
        X_val, y_val, _ = extract_segments(*val, hop)
        input_shape = X_train.shape[1:]
        tf.keras.backend.clear_session()

        tuner = kt.RandomSearch(
            lambda hp: build_tunable_model(hp, input_shape, num_classes),
            objective='val_accuracy',
            max_trials=max_trials,
            seed=SEED,
            directory=os.path.join(output_dir, 'tuning'),
            project_name=f'hop_{hop}_tracksplit',
        )
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[early_stop], verbose=1)

        val_acc = tuner.oracle.get_best_trials(num_trials=1)[0].score
        val_acc_per_hop[hop] = val_acc
        if val_acc > best_val:
            best_val = val_acc
            best_config = {'hop_length': hop, **tuner.get_best_hyperparameters(num_trials=1)[0].values}

    for hop, acc in val_acc_per_hop.items():
        print(f"Hop length {hop}: best validation accuracy {acc * 100:.2f}%")
    return best_config


def evaluate(model, test, num_classes, hop):
    """Segment-level and track-level (soft voting) accuracy on the test tracks."""
    p_test, l_test = test
    X_test, y_test, t_test = extract_segments(p_test, l_test, hop)
    probs = model.predict(X_test, verbose=0)

    track_probs = np.zeros((len(p_test), num_classes))
    np.add.at(track_probs, t_test, probs)
    track_probs /= np.bincount(t_test, minlength=len(p_test))[:, None]

    return {
        'segment_accuracy': float(np.mean(probs.argmax(axis=1) == y_test)),
        'track_accuracy': float(np.mean(track_probs.argmax(axis=1) == l_test)),
        'n_test_segments': int(len(y_test)),
        'n_test_tracks': int(len(l_test)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default=os.environ.get('DATA_DIR', 'data/genres'))
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--config', help='existing best_config.json; skips the search')
    parser.add_argument('--max-trials', type=int, default=20)
    parser.add_argument('--tune-epochs', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()

    set_seeds(SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    genres = list_genres(args.data_dir)
    train, val, test = split_tracks(*list_tracks(args.data_dir, genres), seed=SEED)
    print(f"Tracks - train: {len(train[0])}, val: {len(val[0])}, test: {len(test[0])}")

    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = tune(train, val, len(genres), args.output_dir, args.max_trials, args.tune_epochs)
    config['genres'] = genres
    with open(os.path.join(args.output_dir, 'best_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    hop = config['hop_length']
    X_train, y_train, _ = extract_segments(*train, hop)
    X_val, y_val, _ = extract_segments(*val, hop)

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    model = build_cnn(config, X_train.shape[1:], len(genres))
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=args.epochs, callbacks=[early_stop])
    model.save(os.path.join(args.output_dir, 'final_model.keras'))

    # First and only use of the test set
    results = evaluate(model, test, len(genres), hop)
    print(f"Segment-level test accuracy: {results['segment_accuracy'] * 100:.2f}%")
    print(f"Track-level test accuracy:   {results['track_accuracy'] * 100:.2f}%")
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
