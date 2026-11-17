"""Predict the genre of full songs with soft voting (average of segment probabilities).

    python src/predict.py --model outputs/final_model.keras --config outputs/best_config.json song.mp3
    python src/predict.py --model ... --config ... --eval-dir "test examples"   # one folder per genre
"""
import argparse
import json
import os

import librosa
import numpy as np
import tensorflow as tf

from features import GTZAN_GENRES, SAMPLE_RATE, list_audio_files, signal_to_segments


def predict_file(path, model, hop_length):
    """Return the averaged class probabilities over all full 3 s segments of the file."""
    signal, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    X = signal_to_segments(signal, hop_length)
    if len(X) == 0:
        raise ValueError(f"{path}: shorter than one 3 s segment")
    return model.predict(X, verbose=0).mean(axis=0)


def evaluate_folder(eval_dir, model, genres, hop_length):
    results = []
    for folder in sorted(os.listdir(eval_dir)):
        true_genre = folder.lower().strip()
        folder_path = os.path.join(eval_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for f in list_audio_files(folder_path):
            probs = predict_file(os.path.join(folder_path, f), model, hop_length)
            predicted = genres[int(np.argmax(probs))]
            results.append({'filename': f, 'true_genre': true_genre, 'predicted_genre': predicted,
                            'confidence': float(probs.max())})
            print(f"{'OK ' if predicted == true_genre else 'X  '} {true_genre:>9} -> {predicted:<9} {f}")

    correct = sum(r['true_genre'] == r['predicted_genre'] for r in results)
    print(f"\nAccuracy: {correct}/{len(results)} = {correct / len(results) * 100:.2f}%")
    return {'accuracy': correct / len(results), 'correct': correct, 'total': len(results), 'songs': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='*', help='audio files to classify')
    parser.add_argument('--model', default='outputs/final_model.keras')
    parser.add_argument('--config', default='outputs/best_config.json')
    parser.add_argument('--eval-dir', help='folder with one sub-folder of songs per genre')
    parser.add_argument('--out', help='write evaluation results to this JSON file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    model = tf.keras.models.load_model(args.model)
    # The notebook's best_config.json has no genre list; GTZAN folder order is alphabetical
    genres = config.get('genres', GTZAN_GENRES)

    for path in args.files:
        probs = predict_file(path, model, config['hop_length'])
        top = np.argsort(probs)[::-1][:3]
        print(f"{os.path.basename(path)}: " + ", ".join(f"{genres[i]} {probs[i] * 100:.1f}%" for i in top))

    if args.eval_dir:
        results = evaluate_folder(args.eval_dir, model, genres, config['hop_length'])
        if args.out:
            with open(args.out, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
