from statsmodels.tsa.arima.model import ARIMA
import warnings, time
import numpy as np

class FrozenARIMABaseline:
    """
    Fit once on training data. During streaming inference, update the filtered
    state with new observations without refitting parameters.
    """
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model_fit = None

    def train(self, train_series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(train_series, order=self.order)
            self.model_fit = model.fit()

    def predict_point(self, new_obs=None):
        """
        Predict next step based on current state.
        If new_obs is provided, update internal state AFTER forecasting (typical streaming).
        """
        if self.model_fit is None:
            raise RuntimeError("Call train() first.")

        start = time.perf_counter()

        # 1-step ahead forecast from current state
        yhat = float(self.model_fit.forecast(steps=1)[0])

        # Update state with the newly observed value (no parameter refit)
        if new_obs is not None:
            # append updates the state; refit=False keeps parameters fixed
            self.model_fit = self.model_fit.append([new_obs], refit=False)

        latency = time.perf_counter() - start
        return yhat, latency
