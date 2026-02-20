import torch
import torch.nn as nn
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        return self.linear(lstm_out[:, -1, :])

# 2. Reusable Benchmarking Wrapper
class LSTMBaseline:
    def __init__(self, seq_length=50):
        self.model = LSTMModel()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.seq_length = seq_length
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, data, epochs=20):
        """Standard batch training on the first 20% of data"""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = self._create_sequences(scaled_data)
        
        self.model.train()
        start = time.perf_counter()
        for i in range(epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self.optimizer.step()
        return time.perf_counter() - start

    
    
    def predict_point(self, window):
        """Perform single-step inference and measure latency."""
        self.model.eval()
        
        # 1. Prepare and scale the input window
        scaled_window = self.scaler.transform(np.array(window).reshape(-1, 1))
        
        # 2. Reshape to (Batch, Seq, Feature) -> (1, 50, 1)
        input_tensor = torch.tensor(scaled_window).float().unsqueeze(0)
        
        # 3. Measure inference latency
        start_time = time.perf_counter()
        with torch.no_grad():
            pred_scaled = self.model(input_tensor).item()
        latency = time.perf_counter() - start_time
        
        # 4. Invert scaling to get actual value
        pred_val = self.scaler.inverse_transform([[pred_scaled]])[0][0]
        
        return pred_val, latency

    def _create_sequences(self, data):
        xs, ys = [], []
        # 'data' is already (N, 1) from MinMaxScaler.fit_transform
        for i in range(len(data) - self.seq_length):
            # data[i:(i + self.seq_length)] is shape (50, 1)
            xs.append(data[i:(i + self.seq_length)])
            ys.append(data[i + self.seq_length])
        
        # Convert lists of arrays to a single numpy array first for speed
        # X will be shape (N_samples, 50, 1)
        X = np.array(xs)
        y = np.array(ys)
        
        # Convert to tensors. Since X is already 3D (Batch, Seq, Feature), 
        # do NOT use unsqueeze(-1) unless your data was 1D.
        return torch.tensor(X).float(), torch.tensor(y).float()