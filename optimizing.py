import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch
import os

# --- 1. Load the data and pre-trained models ---

# Define the path to your new data file
parquet_file_path = '/run/media/theodoros/E/projects/hackathone/train-00000-of-00001-9564e8b05b4757ab.parquet'

# Load the trained Keras model
model_path = "/run/media/theodoros/E/projects/hackathone/prompt_injection_model_lstm.keras"
model_lstm = tf.keras.models.load_model(model_path)

# Load the multilingual-e5-base for embeddings
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
model_e5 = AutoModel.from_pretrained('intfloat/multilingual-e5-base')
device = "cpu" if torch.cuda.is_available() else "cpu"
model_e5.to(device)

# --- 2. Create the embedding function ---

def get_embeddings(texts):
    texts_with_prefix = [f"query: {text}" for text in texts]
    batch_dict = tokenizer(
        texts_with_prefix, 
        max_length=512, 
        padding=True, 
        truncation=True, 
        return_tensors='pt'
    )
    
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
    
    with torch.no_grad():
        outputs = model_e5(**batch_dict)
    
    embeddings = F.normalize(outputs.last_hidden_state.mean(dim=1), p=2, dim=1)
    return embeddings.cpu().numpy()

# --- 3. Process the new data ---

print("Processing new parquet data...")
df_new = pd.read_parquet(parquet_file_path)
X_new = df_new['text'].values
y_new = df_new['label'].values

# Generate embeddings for the new data
X_new_embeddings = get_embeddings(X_new.tolist())

# Get raw prediction probabilities from your trained model
y_pred_probs = model_lstm.predict(X_new_embeddings).flatten()

# --- 4. Find the best threshold ---

thresholds = np.arange(0.0, 1.0, 0.01)
best_f1_score = 0
best_threshold = 0

print("Searching for the optimal threshold...")
for threshold in thresholds:
    # Convert probabilities to binary predictions based on the current threshold
    y_pred_binary = (y_pred_probs > threshold).astype(int)
    
    # Calculate the F1-score for this threshold
    f1 = f1_score(y_new, y_pred_binary)
    
    # Update the best score and threshold if the current one is better
    if f1 > best_f1_score:
        best_f1_score = f1
        best_threshold = threshold

print("\n--- Results ---")
print(f"Optimal Threshold: {best_threshold:.2f}")
print(f"Best F1-Score: {best_f1_score:.4f}")