import numpy as np
import pandas as pd
import time
import warnings
from statsmodels.tsa.arima.model import ARIMA

class StaticARIMABaseline:
    """
    A high-performance ARIMA implementation designed for real-time inference.
    
    This class follows a 'Train-Once, Predict-Many' philosophy. It fits model 
    coefficients on an initial training set and then freezes them. For new 
    data, it uses the .apply() method to update the internal state without 
    re-optimizing parameters, maximizing throughput.
    """

    def __init__(self, order=(5, 1, 0)):
        """
        Initializes the ARIMA model parameters.

        Args:
            order (tuple): The (p, d, q) parameters for the ARIMA model.
                           p: autoregressive lags, d: differencing, q: moving average.
        """
        self.order = order
        self.model_fit = None

    def train(self, train_data):
        """
        Fits the ARIMA model to find optimal coefficients.

        This should be called on the initial 10-20% of your signal. It performs
        the computationally expensive Maximum Likelihood Estimation (MLE) to 
        identify the data's pattern.

        Args:
            train_data (np.array or pd.Series): The historical data to train on.
        """
        with warnings.catch_warnings():
            # ARIMA fitting can be noisy with convergence warnings; 
            # we ignore them here as this is a baseline.
            warnings.simplefilter("ignore")
            model = ARIMA(train_data, order=self.order)
            self.model_fit = model.fit()

    def predict_point(self, window):
        """
        Generates a one-step-ahead forecast using frozen parameters.

        Instead of retraining, this 'applies' the previously learned weights 
        to the most recent window of data. This is significantly faster 
        than standard ARIMA forecasting.

        Args:
            window (np.array): The most recent 'p' values needed for prediction.

        Returns:
            tuple: (predicted_value, inference_latency)
        """
        start_time = time.perf_counter()
        
        # .apply(window) creates a new Results object using the same 
        # coefficients but updated with the latest observations.
        updated_results = self.model_fit.apply(window)
        
        # Forecast the very next point
        yhat = updated_results.forecast(steps=1)[0]
        
        latency = time.perf_counter() - start_time
        return yhat, latency