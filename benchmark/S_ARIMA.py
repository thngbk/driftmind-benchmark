from statsmodels.tsa.arima.model import ARIMA
import warnings
import time
class StaticARIMABaseline:
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model_fit = None

    def train(self, train_data):
        """Fit once on the 20% training block."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(train_data, order=self.order)
            self.model_fit = model.fit()

    def predict_point(self, window):
        """Ultra-fast prediction using the fixed coefficients."""
        start_time = time.perf_counter()
        
        # Apply fixed parameters to the new data window
        # This uses the 'apply' method to avoid re-optimizing
        updated_model = self.model_fit.apply(window)
        yhat = updated_model.forecast(steps=1)[0]
        
        latency = time.perf_counter() - start_time
        return yhat, latency