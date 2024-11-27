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
