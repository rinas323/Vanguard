# model_creation.py
import torch
import torch.nn as nn


class FloodLandslideLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(FloodLandslideLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM expects input shape: (batch, time_steps, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        # Final layer to map the hidden state to our two outputs
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Use the output from the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        # For multi-label binary classification, apply sigmoid activation
        return torch.sigmoid(out)

def save_model(model, path):
    """Save the model state_dict to a file."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_class, path, *model_args, **model_kwargs):
    """Load the model state_dict from a file and return the model in eval mode."""
    model = model_class(*model_args, **model_kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

if __name__ == "__main__":
    # Example: Create a model and save it
    input_size = 4
    hidden_size = 64
    num_layers = 2
    output_size = 2
    model = FloodLandslideLSTM(input_size, hidden_size, num_layers, output_size)
    print(model)
    save_model(model, "models/flood_landslide_model.pth")
