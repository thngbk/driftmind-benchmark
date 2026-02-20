import warnings
import time
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA

class TriggeredARIMABaseline:
    """
    ARIMA implementation that separates state updates from parameter fitting.
    """
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model_fit = None

    def train(self, window):
        """Force a full parameter re-fit on the provided window."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(window, order=self.order)
            # Use previous params to help the optimizer converge faster
            start_params = self.model_fit.params if self.model_fit else None
            self.model_fit = model.fit(start_params=start_params)

    def predict_point(self, window):
        """Fast state update and 1-step forecast using FIXED parameters."""
        if self.model_fit is None:
            self.train(window)
        
        start_time = time.perf_counter()
        # .apply updates the Kalman Filter state for the window without refitting coefficients
        updated_model = self.model_fit.apply(window)
        yhat = float(updated_model.forecast(steps=1)[0])
        
        latency = time.perf_counter() - start_time
        return yhat, latency