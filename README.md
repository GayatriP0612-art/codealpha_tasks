# AI Mini-Projects

This repository contains four standalone Python applications, each completing one task:

1. **Language Translation Tool** — `translation_tool.py`
2. **FAQ Chatbot** — `faq_chatbot.py`
3. **Music Generation with AI** — `music_generator.py`
4. **Object Detection and Tracking** — `object_tracker.py`

Each script is self-contained and can be run independently. See the sections below for setup and usage.

---

## 1. Language Translation Tool (`translation_tool.py`)

A desktop GUI app to translate text between languages.

**Features**
- Tkinter UI to enter text and select source & target languages
- Translation via Google Translate (through the `deep-translator` library)
- Displays the translated text
- Copy-to-clipboard button
- Text-to-speech playback (optional, via `pyttsx3`)

**Setup**
```bash
pip install deep-translator pyttsx3
```

**Run**
```bash
python translation_tool.py
```

---

## 2. FAQ Chatbot (`faq_chatbot.py`)

A chatbot that answers user questions by matching them against a set of FAQs.

**Features**
- Sample FAQ dataset (editable `FAQS` list at the top of the file — question/answer pairs)
- NLP preprocessing with NLTK (tokenization, lowercasing, stopword removal, lemmatization)
- Matches user questions to the closest FAQ using TF-IDF + cosine similarity (scikit-learn)
- Displays the best matching answer, with a fallback message if no good match is found
- Simple Tkinter chat window UI (optional console mode also available)

**Setup**
```bash
pip install nltk scikit-learn
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

**Run**
```bash
python faq_chatbot.py            # chat window
python faq_chatbot.py --console  # console mode
```

---

## 3. Music Generation with AI (`music_generator.py`)

Trains an LSTM neural network on MIDI files and generates new music sequences.

**Features**
- Parses a folder of MIDI files into note/chord sequences using `music21`
- Builds a 3-layer LSTM model (Keras/TensorFlow) to learn musical patterns
- Trains on the extracted sequences, saving the best model checkpoint
- Generates new note sequences from the trained model and converts them back into a playable `.mid` file

**Setup**
```bash
pip install music21 tensorflow numpy
```
You'll need a folder of `.mid` files to train on (e.g. classical or jazz MIDI datasets from Kaggle). Place them in a folder such as `./midi_songs/`.

**Run**
```bash
# Train
python music_generator.py --train --data_dir ./midi_songs --epochs 100

# Generate (after training)
python music_generator.py --generate --output generated_output.mid --length 300
```

**Note:** Training is computationally intensive; a GPU-enabled TensorFlow setup is recommended for larger datasets/epoch counts.

---

## 4. Object Detection and Tracking (`object_tracker.py`)

Real-time object detection and multi-object tracking on a webcam or video file.

**Features**
- Real-time video input via OpenCV (webcam or video file)
- Object detection using a pre-trained YOLOv8 model (`ultralytics`)
- Per-frame bounding box + class label extraction
- Multi-object tracking using a custom SORT implementation (Kalman filter + Hungarian/IoU assignment)
- Live display of bounding boxes, class labels, and persistent tracking IDs
- Optional saving of the annotated output video

**Setup**
```bash
pip install opencv-python ultralytics numpy scipy filterpy
```
The first run automatically downloads the YOLOv8n weights (`yolov8n.pt`, ~6MB) — internet access is required once.

**Run**
```bash
# Webcam
python object_tracker.py --source 0

# Video file, save annotated output
python object_tracker.py --source path/to/video.mp4 --output tracked.mp4
```
Press `q` to close the live window.

---

## Requirements Summary

| Task | Key Libraries |
|------|---------------|
| Translation Tool | `deep-translator`, `pyttsx3`, `tkinter` |
| FAQ Chatbot | `nltk`, `scikit-learn`, `tkinter` |
| Music Generation | `music21`, `tensorflow`, `numpy` |
| Object Detection & Tracking | `opencv-python`, `ultralytics`, `scipy`, `filterpy` |

Install everything at once:
```bash
pip install deep-translator pyttsx3 nltk scikit-learn music21 tensorflow numpy opencv-python ultralytics scipy filterpy
```

## Project Structure
```
.
├── translation_tool.py
├── faq_chatbot.py
├── music_generator.py
├── object_tracker.py
└── README.md
```
