# training.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model_creation import FloodLandslideLSTM, save_model
import pandas as pd  # Import pandas for CSV reading

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

def load_data(data_folder="dataset"):
    # This function loads from .npy files (not used for CSV)
    X_path = os.path.join(data_folder, "X.npy")
    y_path = os.path.join(data_folder, "y.npy")
    X = np.load(X_path)
    y = np.load(y_path)
    print("Input shape:", X.shape)
    print("Labels shape:", y.shape)
    return X, y

def load_data_from_csv(csv_path, time_steps=30, num_features=4):
    """
    Load data from a CSV file with the following structure:
    id, rainfall_day1, soiltype_day1, wind_day1, geostruct_day1, 
        ... repeated for each day (total of time_steps days) ...,
    flood_prone, landslide_prone
    """
    df = pd.read_csv(csv_path)
    
    # If there's an id column, drop it
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

def train_model(model, dataloader, num_epochs=50, learning_rate=0.001):
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataset_size = len(dataloader.dataset)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        avg_loss = epoch_loss / dataset_size
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return model

if __name__ == "__main__":
    # -------------------------------
    # Choose your data source:
    # Option A: Load from .npy files (comment out if using CSV)
    # X, y = load_data("dataset")
    
    # Option B: Load from a CSV file (uncomment this block)
    csv_path = "dataset/training_data.csv"  # Specify the CSV file path here
    X, y = load_data_from_csv(csv_path, time_steps=3, num_features=4)
    # -------------------------------
    
    batch_size = 16
    dataset = FloodLandslideDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create and train the model
    input_size = X.shape[2]  # Number of features per time step
    hidden_size = 64
    num_layers = 2
    output_size = 2  # Two outputs: [flood_prone, landslide_prone]
    model = FloodLandslideLSTM(input_size, hidden_size, num_layers, output_size)
    
    model = train_model(model, dataloader, num_epochs=50, learning_rate=0.001)
    
    # Save the trained model
    save_model(model, "models/flood_landslide_model.pth")
