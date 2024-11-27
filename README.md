# Voice Command Recognition with Spectrogram and CNN

This project implements a voice command recognition system using spectrograms and a Convolutional Neural Network (CNN). The system is capable of classifying audio commands into predefined categories by processing their spectrogram representations.

---

## Project Overview

The workflow includes:
1. **Data Preparation:**
   - Audio data is downloaded, preprocessed, and split into training, validation, and testing datasets.
   - Commands are converted into spectrograms using Short-Time Fourier Transform (STFT).
2. **Data Visualization:**
   - Both raw waveforms and their spectrograms are visualized.
3. **Model Architecture:**
   - A CNN model with normalization, convolutional layers, and dropout is trained to classify spectrograms.
4. **Evaluation:**
   - The model's performance is evaluated using confusion matrices, accuracy metrics, and loss curves.

---

## Key Features

- Utilizes **TensorFlow** and **Keras** for machine learning.
- Converts audio signals into spectrograms for robust feature extraction.
- Implements a **CNN model** to classify spectrograms.
- Includes real-time waveform and spectrogram visualization.
- Displays model performance metrics with **loss and accuracy curves**.
- Provides a confusion matrix for detailed performance analysis.

  ---
  
## Installation

### Dependencies
Ensure you have Python installed along with the following packages:
- `tensorflow`
- `numpy`
- `matplotlib`
- `seaborn`
- `pathlib`
- `IPython`

You can install these dependencies using:
```python
import os
import pathlib
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Resizing, Input, Normalization
from keras.models import Sequential
from IPython import display
import matplotlib.pyplot as plt
%matplotlib widget
```
---

## Setup

1. Clone this repository:
```bash
git clone https://github.com/MohammadHosseinDalghi/Voice-Processing-With-CNN.git
cd Voice-Processing-With-CNN
```
2. Run the script:
```bash
python index.ipynb
```

---

## Dataset
You can download the dataset using:
```python
DATASET_PATH = 'data/'

data_dir = pathlib.Path(DATASET_PATH)

tf.keras.utils.get_file(
    'voicedataset.zip',
    origin='http://aiolearn.com/dl/datasets/voicedata.zip',
    extract=True,
    cache_dir='.',
    cache_subdir='data'
)

print('DONE!')
```

---

## Model Details

* Input Shape: Spectrograms resized to `32 × 32`.
* Key Layers:
   - Two `onv2D` layers with ReLU activation.
   - `MaxPooling2D` for spatial dimensionality reduction.
   - `Dropout` for overfitting prevention.
   - `Dense` layers for final predictions.
* Loss Function: Sparse Categorical Crossentropy.
* Optimizer: Adam.

---

## Results
* High accuracy achieved in classifying commands.
* Comprehensive evaluation using metrics like:
   - Training Loss
   - Validation Loss
   - Test Accuracy
* Visualization of model predictions and performance.
