# evaluate.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from model_creation import FloodLandslideLSTM, load_model

# Define the dataset class (same as in training)
class FloodLandslideDataset(Dataset):
    def __init__(self, X, y):
        # X: numpy array of shape (num_samples, time_steps, num_features)
        # y: numpy array of shape (num_samples, 2)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data_from_csv(csv_path, time_steps=3, num_features=4):
    """
    Load data from a CSV file with the following structure:
    id, rainfall_day1, soiltype_day1, wind_day1, geostruct_day1, 
        ... repeated for each day ...,
    flood_prone, landslide_prone
    """
    df = pd.read_csv(csv_path)
    
    # Drop the 'id' column if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    total_feature_columns = time_steps * num_features
    features = df.iloc[:, :total_feature_columns].values
    labels = df.iloc[:, total_feature_columns:].values
    
    # Reshape features to (num_samples, time_steps, num_features)
    features = features.reshape(-1, time_steps, num_features)
    
    print("Loaded CSV - Features shape:", features.shape)
    print("Loaded CSV - Labels shape:", labels.shape)
    return features, labels

def evaluate_model(model, dataloader):
    """
    Evaluate the model on the test data and print the accuracy.
    For multi-label binary classification, we'll compute the accuracy
    as the percentage of correctly predicted binary labels over all labels.
    """
    model.eval()
    total_labels = 0
    correct_labels = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            # Convert outputs (probabilities) to binary predictions using threshold 0.5
            predictions = (outputs > 0.5).float()
            total_labels += y_batch.numel()
            correct_labels += (predictions == y_batch).sum().item()
    accuracy = 100 * correct_labels / total_labels
    print(f"Overall Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    # Specify the CSV file path for testing data.
    test_csv_path = "dataset/testing_data.csv"  # Adjust as needed.
    
    # Adjust these parameters to match your CSV structure:
    time_steps = 3     # For example, 3 days of test data
    num_features = 4   # e.g., rainfall, soil type, wind, geostruct
    
    # Load test data from CSV
    X_test, y_test = load_data_from_csv(test_csv_path, time_steps, num_features)
    
    # Create dataset and dataloader for test data
    batch_size = 16
    test_dataset = FloodLandslideDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model parameters should match those used during training.
    input_size = num_features
    hidden_size = 64
    num_layers = 2
    output_size = 2  # [flood_prone, landslide_prone]
    
    # Load the trained model. (Ensure the path matches where your model is saved.)
    model_path = "models/flood_landslide_model.pth"
    model = load_model(FloodLandslideLSTM, model_path, input_size, hidden_size, num_layers, output_size)
    
    # Evaluate the model on test data
    evaluate_model(model, test_dataloader)
