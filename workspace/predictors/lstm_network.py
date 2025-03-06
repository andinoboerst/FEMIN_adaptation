import numpy as np
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

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
        self.trainY = torch.tensor(self.Y_train, dtype=torch.float32)

        # Assuming your input data is a NumPy array or a list of NumPy arrays
        # Example input_data: list of numpy arrays.

        # Initialize the scaler
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        # Combine all training simulations into a single NumPy array (for fitting)
        combined_input_data = np.concatenate(X, axis=0)
        combined_target_data = np.concatenate(Y, axis=0)

        self.input_scaler.fit(combined_input_data)
        self.output_scaler.fit(combined_target_data)

        # middle_sim = int(len(X) / 2)
        # self.input_scaler.fit(X[middle_sim])
        # self.output_scaler.fit(Y[middle_sim])

        input_data = [torch.tensor(self.input_scaler.transform(x), dtype=torch.float32) for x in X]
        target_data = [torch.tensor(self.output_scaler.transform(y), dtype=torch.float32) for y in Y]

        # with open("in_data.npy", "wb") as f:
        #     np.save(f, np.array(input_data))

        # with open("out_data.npy", "wb") as f:
        #     np.save(f, np.array(target_data))

        # Assuming your input_data and target_data are lists of tensors
        dataset = SimulationDataset(input_data, target_data)
        batch_size = input_data[0].size(0)  # Adjust as needed
        self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # shuffle is false

        # Hyperparameters
        self.input_size = input_data[0].size(1)  # Number of features in your simulation
        self.hidden_size = 128
        self.output_size = target_data[0].size(1)  # Number of outputs to predict
        self.num_layers = 2
        self.learning_rate = 0.01
        self.num_epochs = 20
        self.num_timesteps = input_data[0].size(0)  # All simulations have same length

    def fit(self) -> None:
        # Model, loss, and optimizer
        self._model = LSTMModel(self.input_size, self.hidden_size, self.output_size, self.num_layers)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            for batch_inputs, batch_targets in self.train_loader:
                self._model.train()
                optimizer.zero_grad()

                # Initialize hidden state for the batch
                hidden = None

                # Iterate through timesteps
                for t in range(self.num_timesteps):
                    # Get the input and target for the current timestep
                    batch_input_t = batch_inputs[:, t:t + 1, :]
                    batch_target_t = batch_targets[:, t, :]

                    # Forward pass
                    outputs, hidden = self._model(batch_input_t, hidden)

                    # Compute loss
                    loss = criterion(outputs.squeeze(1), batch_target_t)

                    # Backward pass and optimization
                    loss.backward(retain_graph=True)  # retain graph is very important.
                    optimizer.step()
                    optimizer.zero_grad()  # clear gradients every timestep.

                    for h in hidden:
                        h.detach_()

            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.8f}')

            self.save("current_lstm_model.pkl")

        print('Training finished!')
        self._model.eval()

    def initialize_memory_variables(self) -> None:
        self.hidden = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_normalized = self.input_scaler.transform(x.reshape(1, -1))

        x_tensor = torch.tensor(x_normalized, dtype=torch.float32)
        with torch.no_grad():
            predicted, self.hidden = self._model(x_tensor[None, :, :], self.hidden)

        predicted_normalized = self.output_scaler.inverse_transform(predicted.detach().numpy()[0].reshape(1, -1))

        return predicted_normalized


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
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.layer_size = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        # If hidden and cell states are not provided, initialize them as zeros
        if hidden is None:
            batch_size = x.size(0)
            h0 = torch.zeros(self.layer_size, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.layer_size, batch_size, self.hidden_size).to(x.device)
            # h0 = torch.zeros(self.layer_size, self.hidden_size).to(x.device)
            # c0 = torch.zeros(self.layer_size, self.hidden_size).to(x.device)
            # if batch_size == 1:
            #     hidden = (h0[:, 0, :], c0[:, 0, :])
            # else:
            hidden = (h0, c0)

        # Forward pass through LSTM
        out, hidden_n = self.lstm(x, hidden)
        # out = self.fc(out[:, -1, :])  # Selecting the last output
        out = self.fc(out)  # Selecting the last output
        return out, hidden_n


class SimulationDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]
