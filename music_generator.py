#!/usr/bin/env python3
"""
Music Generation with AI
-------------------------
Trains an LSTM neural network on a folder of MIDI files and generates new
MIDI music sequences from the learned patterns.

Pipeline:
1. Collect MIDI data       -> put .mid / .midi files in a folder (e.g. ./midi_songs)
2. Preprocess               -> music21 extracts notes/chords into sequences
3. Build deep learning model-> LSTM network (Keras/TensorFlow)
4. Train the model          -> learns note-sequence patterns
5. Generate + convert       -> samples new note sequences, writes a playable .mid file

Setup (run once in a terminal):
    pip install music21 tensorflow numpy

You also need a folder of MIDI files to train on. A good free source:
    https://www.kaggle.com/datasets (search "classical midi" or "jazz midi")
Place the .mid files in a folder, e.g.:  ./midi_songs/

Usage:
    # 1) Train a model on your MIDI folder (creates notes.pkl + model .keras file)
    python music_generator.py --train --data_dir ./midi_songs --epochs 100

    # 2) Generate a new piece from the trained model
    python music_generator.py --generate --output generated_output.mid

    # Optional: change how many notes the generated piece has
    python music_generator.py --generate --length 300
"""

import argparse
import glob
import pickle
import os
import sys

import numpy as np

try:
    from music21 import converter, instrument, note, chord, stream
except ImportError:
    raise SystemExit("Missing dependency 'music21'.\nInstall it with:  pip install music21")

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.utils import to_categorical
except ImportError:
    raise SystemExit(
        "Missing dependency 'tensorflow'.\nInstall it with:  pip install tensorflow"
    )


SEQUENCE_LENGTH = 100
NOTES_FILE = "notes.pkl"
MODEL_FILE = "music_model.keras"


# ---------------------------------------------------------------------------
# 1 & 2. Collect + preprocess MIDI data into note/chord sequences
# ---------------------------------------------------------------------------
def extract_notes(data_dir: str):
    """Parse every MIDI file in data_dir into a flat list of note/chord tokens."""
    midi_files = glob.glob(os.path.join(data_dir, "*.mid")) + glob.glob(
        os.path.join(data_dir, "*.midi")
    )
    if not midi_files:
        raise SystemExit(
            f"No .mid/.midi files found in '{data_dir}'.\n"
            "Add some MIDI files there first (see script docstring for sources)."
        )

    notes = []
    print(f"Found {len(midi_files)} MIDI file(s). Parsing...")

    for i, file in enumerate(midi_files, 1):
        print(f"  [{i}/{len(midi_files)}] {os.path.basename(file)}")
        try:
            midi = converter.parse(file)
        except Exception as exc:
            print(f"    Skipped (could not parse): {exc}")
            continue

        try:
            parts = instrument.partitionByInstrument(midi)
            elements = parts.parts[0].recurse() if parts else midi.flat.notes
        except Exception:
            elements = midi.flat.notes

        for el in elements:
            if isinstance(el, note.Note):
                notes.append(str(el.pitch))
            elif isinstance(el, chord.Chord):
                notes.append(".".join(str(n) for n in el.normalOrder))

    with open(NOTES_FILE, "wb") as f:
        pickle.dump(notes, f)

    print(f"Extracted {len(notes)} note/chord events. Saved to {NOTES_FILE}")
    return notes


def prepare_sequences(notes, sequence_length=SEQUENCE_LENGTH):
    """Turn the flat note list into (X, y) training sequences for the LSTM."""
    pitch_names = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(pitch_names)}
    n_vocab = len(pitch_names)

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        seq_in = notes[i : i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    X = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(n_vocab)
    y = to_categorical(network_output, num_classes=n_vocab)

    return X, y, pitch_names, n_vocab, note_to_int


# ---------------------------------------------------------------------------
# 3. Build the LSTM model
# ---------------------------------------------------------------------------
def build_model(input_shape, n_vocab):
    model = Sequential([
        LSTM(256, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dense(256),
        Dropout(0.3),
        Dense(n_vocab),
        Activation("softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
def train(data_dir: str, epochs: int, batch_size: int):
    notes = extract_notes(data_dir)
    X, y, pitch_names, n_vocab, _ = prepare_sequences(notes)

    print(f"Vocabulary size: {n_vocab} unique notes/chords")
    print(f"Training sequences: {len(X)}")

    model = build_model((X.shape[1], X.shape[2]), n_vocab)
    model.summary()

    checkpoint = ModelCheckpoint(
        MODEL_FILE, monitor="loss", verbose=1, save_best_only=True, mode="min"
    )

    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
    model.save(MODEL_FILE)
    print(f"\nTraining complete. Model saved to {MODEL_FILE}")


# ---------------------------------------------------------------------------
# 5. Generate new sequences + convert to MIDI
# ---------------------------------------------------------------------------
def generate(output_path: str, length: int):
    if not os.path.exists(NOTES_FILE):
        raise SystemExit(f"'{NOTES_FILE}' not found. Run --train first.")
    if not os.path.exists(MODEL_FILE):
        raise SystemExit(f"'{MODEL_FILE}' not found. Run --train first.")

    with open(NOTES_FILE, "rb") as f:
        notes = pickle.load(f)

    _, _, pitch_names, n_vocab, note_to_int = prepare_sequences(notes)
    int_to_note = {i: n for n, i in note_to_int.items()}

    model = load_model(MODEL_FILE)

    # Seed sequence: a random window from the training data
    network_input = []
    for i in range(len(notes) - SEQUENCE_LENGTH):
        seq_in = notes[i : i + SEQUENCE_LENGTH]
        network_input.append([note_to_int[n] for n in seq_in])

    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]

    generated_notes = []
    print(f"Generating {length} notes...")
    for _ in range(length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        idx = np.argmax(prediction)
        generated_notes.append(int_to_note[idx])
        pattern.append(idx)
        pattern = pattern[1:]

    midi_stream = notes_to_midi_stream(generated_notes)
    midi_stream.write("midi", fp=output_path)
    print(f"Saved generated music to '{output_path}'")


def notes_to_midi_stream(generated_notes, step_duration=0.5):
    """Convert a list of note/chord token strings into a music21 Stream."""
    offset = 0
    output_notes = []

    for token in generated_notes:
        if "." in token or token.isdigit():
            # Chord
            chord_notes = []
            for n in token.split("."):
                new_note = note.Note(int(n))
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            # Single note
            new_note = note.Note(token)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += step_duration

    return stream.Stream(output_notes)


# ---------------------------------------------------------------------------
# 6. CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train and generate AI music from MIDI data.")
    parser.add_argument("--train", action="store_true", help="Train the LSTM model")
    parser.add_argument("--generate", action="store_true", help="Generate new music")
    parser.add_argument("--data_dir", default="./midi_songs", help="Folder with training MIDI files")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--output", default="generated_output.mid", help="Output MIDI file path")
    parser.add_argument("--length", type=int, default=200, help="Number of notes to generate")
    args = parser.parse_args()

    if not args.train and not args.generate:
        parser.print_help()
        sys.exit(0)

    if args.train:
        train(args.data_dir, args.epochs, args.batch_size)

    if args.generate:
        generate(args.output, args.length)


if __name__ == "__main__":
    main()
