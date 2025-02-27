import numpy as np
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from predictors.fenicsx_predictor import FenicsxPredictor


# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)


class LSTMNetworkOG(FenicsxPredictor):
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        super().__init__(X, Y, test_size=0.001)

        # Convert data to PyTorch tensors
        self.trainX = torch.tensor(self.X_train, dtype=torch.float32)
        # self.trainY = torch.tensor(self.Y_train[:, -1, :], dtype=torch.float32)
        self.trainY = torch.tensor(self.Y_train, dtype=torch.float32)

        # self.testX = torch.tensor(self.X_test, dtype=torch.float32)
        # self.testY = torch.tensor(self.Y_test[:, -1, :], dtype=torch.float32)

    def fit(self) -> None:
        # Initialize model, loss, and optimizer
        self._model = LSTMModel(input_dim=42, hidden_dim=100, layer_dim=5, output_dim=42)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01)

        # Training loop
        num_epochs = 1000
        h0, c0 = None, None  # Initialize hidden and cell states

        for epoch in range(num_epochs):
            for i in range(1000, self.trainX.shape[1] + 1, 1000):
                self._model.train()
                optimizer.zero_grad()

                # Forward pass
                outputs, h0, c0 = self._model(self.trainX[:, :i, :], h0, c0)

                # Compute loss
                loss = criterion(outputs, self.trainY[:, i - 1, :])
                loss.backward()
                optimizer.step()

                # Detach hidden and cell states to prevent backpropagation through the entire sequence
                h0 = h0.detach()
                c0 = c0.detach()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    def initialize_memory_variables(self) -> None:
        self.h = None
        self.c = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(x[None, None, :], dtype=torch.float32)
        predicted, self.h, self.c = self._model(x_tensor, self.h, self.c)

        print(self.h)
        print(self.c)
        return predicted.detach().numpy()[0]


class LSTMNetwork(FenicsxPredictor):
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        super().__init__(X, Y, test_size=0.001)

        # Convert data to PyTorch tensors
        self.trainX = torch.tensor(self.X_train, dtype=torch.float32)
        # self.trainY = torch.tensor(self.Y_train[:, -1, :], dtype=torch.float32)
        self.trainY = torch.tensor(self.Y_train, dtype=torch.float32)

        # self.trainX = (self.trainX - self.x_mean) / self.x_std
        # self.trainY = (self.trainY - self.y_mean) / self.y_std

        # self.testX = torch.tensor(self.X_test, dtype=torch.float32)
        # self.testY = torch.tensor(self.Y_test[:, -1, :], dtype=torch.float32)

    def fit(self) -> None:
        # Initialize model, loss, and optimizer
        self._model = LSTMModel(input_dim=42, hidden_dim=100, layer_dim=1, output_dim=42)
        criterion = nn.MSELoss()
        # criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01)
        optimizer = torch.optim.SGD(self._model.parameters(), lr=0.01)

        # Training loop
        num_epochs = 1000

        for epoch in range(num_epochs):
            h0, c0 = None, None  # Initialize hidden and cell states
            for i in range(1, self.trainX.shape[1] + 1):
                self._model.train()
                optimizer.zero_grad()

                # Forward pass
                outputs, h0, c0 = self._model(self.trainX[:, i - 1:i, :], h0, c0)

                # Compute loss
                loss = criterion(outputs, self.trainY[:, i - 1, :])
                loss.backward()
                optimizer.step()

                # Detach hidden and cell states to prevent backpropagation through the entire sequence
                h0 = h0.detach()
                c0 = c0.detach()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    def initialize_memory_variables(self) -> None:
        self.h = None
        self.c = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(x[None, None, :], dtype=torch.float32)
        predicted, self.h, self.c = self._model(x_tensor, self.h, self.c)

        print(self.h)
        print(self.c)
        return predicted.detach().numpy()[0]


class LSTMWindowNetwork(FenicsxPredictor):
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        super().__init__(X, Y, test_size=0.001)

        # Convert data to PyTorch tensors
        self.trainX = torch.tensor(self.X_train, dtype=torch.float32)
        self.trainY = torch.tensor(self.Y_train, dtype=torch.float32)

        self.window_size = 5

    def fit(self) -> None:
        # Initialize model, loss, and optimizer
        self._model = LSTMModel(input_dim=42, hidden_dim=100, layer_dim=5, output_dim=42)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01)

        # Training loop
        num_epochs = 1000
        h0, c0 = None, None  # Initialize hidden and cell states

        for epoch in range(num_epochs):
            for i in range(1, self.trainX.shape[1] + 1):
                self._model.train()
                optimizer.zero_grad()

                start_point = max(0, i - self.window_size)
                # Forward pass
                outputs, h0, c0 = self._model(self.trainX[:, start_point:i, :], None, None)

                # Compute loss
                loss = criterion(outputs, self.trainY[:, i - 1:i, :])
                loss.backward()
                optimizer.step()

                # Detach hidden and cell states to prevent backpropagation through the entire sequence
                h0 = h0.detach()
                c0 = c0.detach()

                print(f'Loss: {loss.item():.4f}')

            print('\n')
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    def initialize_memory_variables(self) -> None:
        self.h = None
        self.c = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(x[None, None, :], dtype=torch.float32)
        predicted, self.h, self.c = self._model(x_tensor, self.h, self.c)

        print(self.h)
        print(self.c)
        return predicted.detach().numpy()[0]


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        # If hidden and cell states are not provided, initialize them as zeros
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        # Forward pass through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Selecting the last output
        return out, hn, cn
