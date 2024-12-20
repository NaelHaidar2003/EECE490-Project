# EECE490-Project
Keystroke Dynamics Authentication System

Overview

This project implements a keystroke dynamics authentication system using machine learning techniques. The system analyzes user typing patterns to identify users and evaluate model performance. It includes preprocessing, sequence creation, predictions, and evaluation using a pre-trained LSTM model.

Code Explanation

1. Import Libraries

The script begins by importing necessary Python libraries. These include tools for data manipulation, numerical computations, machine learning preprocessing, evaluation metrics, and visualization. TensorFlow is used for loading and interacting with the pre-trained LSTM model.

2. Load Datasets

Three datasets are loaded: one containing fixed-text typing data, another for free-text typing data, and a demographic dataset. These are used for analysis and model training/testing.

3. Preprocess Fixed-Text Data

Latency features, specifically the downstroke and upstroke times, are extracted from the fixed-text dataset. The data is normalized using MinMaxScaler to ensure all features fall within the same range, which improves the performance and convergence of the machine learning model. The scaler is saved for reuse in future predictions.

4. Create Sequences for LSTM

To train the LSTM model, sequences of a fixed length are created. These sequences represent segments of typing data that the model can analyze to learn patterns over time. Participant IDs are also associated with each sequence to serve as labels for supervised learning.

5. Encode Participant IDs

Participant IDs are converted into numerical labels using a label encoder. This allows the model to process the labels during training. The label encoder is also saved for future use in decoding predictions.

6. Split Data into Train and Test Sets

The processed data is split into training and testing subsets. The training set is used to fit the model, while the test set is used to evaluate its performance and generalization capability.

7. Load the Pre-trained Model

A pre-trained LSTM model is loaded. This model has been trained on similar data and is used to make predictions on the test data without requiring additional training.

8. Make Predictions and Evaluate

The model predicts user identities based on the test sequences. Evaluation metrics such as accuracy, a confusion matrix, and a classification report are generated to measure the model's performance. These metrics provide insight into the model's precision, recall, and F1-score for each participant.

9. Visualize Confusion Matrix

The confusion matrix is visualized using a heatmap to provide an intuitive understanding of the model's performance. This visualization highlights areas where the model correctly or incorrectly predicted user identities.

10. Optional: Test on Free-Text Data

The script includes an optional section for testing on free-text data. This involves normalizing the free-text latencies using the previously saved scaler, creating sequences, and making predictions. The decoded predictions are compared to the true participant IDs for evaluation.


