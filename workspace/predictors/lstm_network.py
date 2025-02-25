import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from predictors.fenicsx_predictor import FenicsxPredictor


# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)


class LSTMNetwork(FenicsxPredictor):
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        super().__init__(X, Y)

        # Convert data to PyTorch tensors
        self.trainX = torch.tensor(self.X_train[:, :, None], dtype=torch.float32)
        self.trainY = torch.tensor(self.Y_train[:, None], dtype=torch.float32)

        self.testX = torch.tensor(self.X_test[:, :, None], dtype=torch.float32)
        self.testY = torch.tensor(self.Y_test[:, None], dtype=torch.float32)

    def fit(self) -> None:
        # Initialize model, loss, and optimizer
        self._model = LSTMModel(input_dim=self.X_base.shape[2], hidden_dim=100, num_layers=1, output_dim=self.X_base.shape[2])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01)

        # Training loop
        num_epochs = 100
        h0, c0 = None, None  # Initialize hidden and cell states

        for epoch in range(num_epochs):
            self._model.train()
            optimizer.zero_grad()

            # Forward pass
            outputs, h0, c0 = self._model(self.trainX, h0, c0)

            # Compute loss
            loss = criterion(outputs, self.trainY)
            loss.backward()
            optimizer.step()

            # Detach hidden and cell states to prevent backpropagation through the entire sequence
            h0 = h0.detach()
            c0 = c0.detach()

            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Predicted outputs
        # model.eval()
        # predicted, _, _ = model(self.testX, h0, c0)

        # # Adjusting the original data and prediction for plotting
        # original = data[seq_length:]  # Original data from the end of the first sequence
        # time_steps = np.arange(seq_length, len(data))  # Corresponding time steps

        # # Plotting
        # plt.figure(figsize=(12, 6))
        # plt.plot(time_steps, original, label='Original Data')
        # plt.plot(time_steps, predicted.detach().numpy(), label='Predicted Data', linestyle='--')
        # plt.title('LSTM Model Predictions vs. Original Data')
        # plt.xlabel('Time Step')
        # plt.ylabel('Value')
        # plt.legend()
        # plt.show()

    def start_prediction_cycle(self) -> None:
        self.h = None
        self.c = None
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        predicted, self.h, self.c = self._model(x, self.h, self.c)

        return predicted





def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        # If hidden and cell states are not provided, initialize them as zeros
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Selecting the last output
        return out, hn, cn
