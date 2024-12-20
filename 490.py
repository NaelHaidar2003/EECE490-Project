import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
fixed_text_data = pd.read_csv("fixed-text.csv")
free_text_data = pd.read_csv("free-text.csv", low_memory=False)
demographics_data = pd.read_csv("demographics.csv")

# Preprocessing fixed-text dataset
latency_columns = [col for col in fixed_text_data.columns if 'D' in col or 'U' in col]
fixed_latencies = fixed_text_data[latency_columns]
participants = fixed_text_data['participant']

# Normalize data
scaler = MinMaxScaler()
fixed_latencies_normalized = scaler.fit_transform(fixed_latencies)
# history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the scaler for later use
import joblib
joblib.dump(scaler, "scaler.pkl")

# Create sequences for LSTM (sequence length = 50)
sequence_length = 50
X_fixed = []
y_fixed = []

for participant_id in participants.unique():
    participant_data = fixed_latencies_normalized[participants == participant_id]
    for i in range(len(participant_data) - sequence_length):
        X_fixed.append(participant_data[i:i + sequence_length])
        y_fixed.append(participant_id)

X_fixed = np.array(X_fixed)

# Encode participant IDs as numeric labels
label_encoder = LabelEncoder()
y_fixed = label_encoder.fit_transform(y_fixed)

# Save the label encoder for later use
joblib.dump(label_encoder, "label_encoder.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_fixed, y_fixed, test_size=0.2, random_state=42)

# Load the trained model
model = tf.keras.models.load_model("keystroke_auth_model.keras")
model.summary()
# Predict on test data
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Decode predictions and true labels
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels_decoded = label_encoder.inverse_transform(y_pred_labels)

# Evaluate accuracy
accuracy = np.mean(y_test == y_pred_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Display predictions for the first 5 samples
print("\nSample Predictions:")
for i in range(5):
    print(f"True User: {y_test_labels[i]} | Predicted User: {y_pred_labels_decoded[i]}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_labels)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
cr = classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_)
print("\nClassification Report:")
print(cr)

# Visualize Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# (Optional) Test on Free-Text Data
latency_columns_free = [col for col in free_text_data.columns if 'D' in col or 'U' in col]
free_latencies = free_text_data[latency_columns_free]
free_latencies_normalized = scaler.transform(free_latencies)

# Create sequences for free-text
X_free = []
y_free = free_text_data['participant']  # True labels for free-text

for participant_id in y_free.unique():
    participant_data = free_latencies_normalized[y_free == participant_id]
    for i in range(len(participant_data) - sequence_length):
        X_free.append(participant_data[i:i + sequence_length])

X_free = np.array(X_free)

# Predict on free-text data
if len(X_free) > 0:
    y_free_pred = model.predict(X_free)
    y_free_pred_labels = np.argmax(y_free_pred, axis=1)
    y_free_pred_labels_decoded = label_encoder.inverse_transform(y_free_pred_labels)

    print("\nFree-Text Predictions (First 5 Samples):")
    for i in range(min(5, len(X_free))):
        print(f"True User: {y_free.iloc[i]} | Predicted User: {y_free_pred_labels_decoded[i]}")
