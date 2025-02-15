# predict.py
import torch
import numpy as np
from model_creation import FloodLandslideLSTM, load_model

def predict_sample(model, sample):
    """
    Predict on a single sample.
    :param model: Trained model.
    :param sample: Numpy array of shape (time_steps, num_features).
    :return: Predicted probabilities as a numpy array.
    """
    model.eval()
    # Add batch dimension: (1, time_steps, num_features)
    sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(sample_tensor)
    return output.squeeze().numpy()

if __name__ == "__main__":
    # Example: Create a random sample for demonstration (adjust dimensions as needed)
    time_steps = 30  # For example, 30 days of data
    num_features = 4
    sample = np.random.rand(time_steps, num_features)
    
    # Load the saved model
    input_size = num_features
    hidden_size = 64
    num_layers = 2
    output_size = 2
    model = load_model(FloodLandslideLSTM, "flood_landslide_model.pth", input_size, hidden_size, num_layers, output_size)
    
    # Predict the outcome for the sample
    prediction = predict_sample(model, sample)
    print("Predicted probabilities [flood, landslide]:", prediction)
