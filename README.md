# Action Detection Using MediaPipe and LSTM

This project implements a real-time human action recognition system using structured body keypoints and sequence-based deep learning.

The implementation is adapted from an educational project by **Nicholas Renotte** and reorganized into a complete, reproducible pipeline covering data collection, preprocessing, training, evaluation, and live inference.

## Overview

The system captures webcam video, extracts pose, face, and hand landmarks using MediaPipe Holistic, and models temporal motion patterns using an LSTM network to classify predefined human actions.

Instead of operating directly on raw video frames, the project focuses on efficient action recognition using numerical keypoint representations.

## Core Components

- MediaPipe Holistic for pose, face, and hand landmark extraction  
- Keypoint preprocessing and NumPy-based sequence storage  
- Fixed-length temporal sequence construction  
- LSTM-based action classification model using TensorFlow/Keras  
- Model evaluation with accuracy metrics and confusion matrices  
- Real-time prediction with live probability visualization  

## Technology Stack

- Python  
- OpenCV  
- MediaPipe  
- TensorFlow / Keras  
- NumPy  
- scikit-learn  
- Matplotlib  

## Workflow

1. Capture video frames from webcam  
2. Extract holistic landmarks per frame  
3. Convert landmarks into numerical keypoint vectors  
4. Form temporal sequences for supervised learning  
5. Train LSTM model on labeled action sequences  
6. Evaluate model performance  
7. Perform real-time action inference  

## Notes

- Model accuracy depends on data quality and consistency during recording.
- Designed as a practical demonstration of sequence modeling on pose data.
- Can be extended to additional actions or alternative temporal models.
