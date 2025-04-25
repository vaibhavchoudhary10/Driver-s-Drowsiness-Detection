# Drowsiness Detection

This project implements a real-time drowsiness detection system using deep learning and computer vision. It uses a trained CNN model to detect whether a person's eyes are open or closed from webcam video, and sounds an alarm if drowsiness is detected.

## Features

- Real-time face and eye detection using OpenCV
- Deep learning model (InceptionV3-based CNN) for eye state classification
- Alarm sound when drowsiness is detected
- Easily trainable on your own dataset

## Project Structure

```
Drowsiness_Detection/
│
├── Models/
│   └── model.h5              # Trained Keras model
├── Prepared_data/            # Training and test images (organized by class)
├── ModeTraining.py           # Model training script
├── main.py                   # Real-time detection script
├── alarm.wav                 # Alarm sound file
├── data_preparation.ipynb    # Data preparation notebook
├── annotation.txt            # (Optional) Data annotation info
├── mrlEyes_2018_01.zip       # (Optional) Example dataset
├── stats_2018_01.ods         # (Optional) Dataset stats
└── venv/                     # Python virtual environment
```

## Requirements

- Python 3.7+
- TensorFlow / Keras
- OpenCV
- Pillow
- pygame
- numpy

Install dependencies with:
```
pip install tensorflow opencv-python pillow pygame numpy
```

## Usage

### 1. Training the Model

If you want to train your own model, organize your data in `Prepared_data/` with subfolders for each class (e.g., `open`, `closed`). Then run:

```
python ModeTraining.py
```

This will train the model and save the best weights to `Models/model.h5`.

### 2. Real-Time Drowsiness Detection

Make sure `Models/model.h5` and `alarm.wav` are present. Then run:

```
python main.py
```

A webcam window will open. If drowsiness is detected (eyes closed for several frames), an alarm will sound.

## Notes

- The model expects images of size 80x80x3.
- You can use the [MRL Eye Dataset](https://mrl.cs.vsb.cz/eyedataset.html) or your own images for training.
- Adjust thresholds and parameters in `main.py` for your use case.
