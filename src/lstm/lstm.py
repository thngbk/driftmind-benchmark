import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


class LSTMModel(nn.Module):
    """
    A standard single-layer LSTM neural network architecture.

    This model processes a sequence of inputs to find non-linear temporal
    relationships. The 'forward' pass takes the final hidden state of the
    LSTM and passes it through a linear head to produce a single-step forecast.
    """

    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        # batch_first=True ensures the input shape is (Batch, Sequence, Feature)
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # We only care about the output of the last time step in the sequence
        lstm_out, _ = self.lstm(input_seq)
        return self.linear(lstm_out[:, -1, :])


class LSTMBaseline:
    """
    A benchmarking wrapper for the LSTM model.

    STRATEGY:
    Uses a Deep Learning approach with data normalization. It learns complex
    patterns through backpropagation during the training phase and performs
    fast tensor-based inference during the streaming phase.

    ANGLE IN EXPERIMENT:
    This represents the 'Non-Linear/Complex Pattern' baseline. Unlike ARIMA,
    which is a linear model, the LSTM can model sophisticated trends and
    seasonality. However, it is a 'black box' and requires more compute power.
    """

    def __init__(self, seq_length=50):
        """
        Initializes the LSTM ecosystem.

        Args:
            seq_length (int): The 'lookback' window (number of previous time
                             steps used to predict the next one).
        """
        self.model = LSTMModel()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.seq_length = seq_length
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, data, epochs=20):
        """
        Trains the neural network on the provided dataset.

        Args:
            data (np.array): Training data.
            epochs (int): Number of complete passes over the training data.

        Note:
            Uses the Adam optimizer and Mean Squared Error (MSE) loss.
            Returns the total training time in seconds.
        """
        # LSTM is sensitive to scale; data must be between 0 and 1
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = self._create_sequences(scaled_data)

        self.model.train()
        start = time.perf_counter()
        for _i in range(epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self.optimizer.step()
        return time.perf_counter() - start

    def predict_point(self, window):
        """
        Performs single-step inference.

        Processes a window of data through the LSTM to predict the next value.
        Includes pre-processing (scaling) and post-processing (inverse scaling).

        Args:
            window (list/array): The sliding window of recent observations.

        Returns:
            tuple: (predicted_value, latency)
        """
        if not hasattr(self.scaler, "scale_"):
            raise RuntimeError(
                "Model is not initialised. Call train() before predict_point()."
            )

        self.model.eval()

        start_time = time.perf_counter()

        # 1. Scaling (must use the scaler fitted during training)
        scaled_window = self.scaler.transform(np.array(window).reshape(-1, 1))

        # 2. Tensor conversion (1, Seq_Length, 1)
        input_tensor = torch.tensor(scaled_window).float().unsqueeze(0)

        # 3. Inference
        with torch.no_grad():  # Disable gradient calculation for speed
            pred_scaled = self.model(input_tensor).item()

        latency = time.perf_counter() - start_time

        # 4. Inverse Scaling to original data units
        pred_val = self.scaler.inverse_transform([[pred_scaled]])[0][0]

        return pred_val, latency

    def _create_sequences(self, data):
        """
        Converts a flat time-series array into overlapping windows (X)
        and targets (y).
        """
        xs, ys = [], []
        for i in range(len(data) - self.seq_length):
            xs.append(data[i : (i + self.seq_length)])
            ys.append(data[i + self.seq_length])

        return torch.tensor(np.array(xs)).float(), torch.tensor(np.array(ys)).float()
