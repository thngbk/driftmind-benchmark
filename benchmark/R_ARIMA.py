from statsmodels.tsa.arima.model import ARIMA
import warnings
import time

class RollingARIMABaseline:
    def __init__(self, order=(5, 1, 0), interval=10):
        self.order = order
        self.interval = interval
        self.model_fit = None
        self.counter = 0

    def predict_point(self, window):
        start_time = time.perf_counter()
        
        # Determine if we need to re-fit or just apply fixed parameters
        if self.model_fit is None or self.counter >= self.interval:
            # Expensive Re-fit
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(window, order=self.order)
                self.model_fit = model.fit()
            self.counter = 0 # Reset interval counter
        
        # Fast Inference using the current fit
        updated_model = self.model_fit.apply(window)
        yhat = updated_model.forecast(steps=1)[0]
        
        self.counter += 1
        latency = time.perf_counter() - start_time
        return yhat, latency