import time
import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA


class StaticARIMABaseline:
    """
    A high-performance ARIMA implementation designed for real-time inference.

    This class follows a 'Train-Once, Predict-Many' philosophy. It fits model
    coefficients on an initial training set and then freezes them. For new
    data, it uses the .apply() method to update the internal state without
    re-optimizing parameters, maximizing throughput.
    """

    _SUPPRESSED = (ConvergenceWarning, UserWarning)

    def __init__(self, order=(5, 1, 0)):
        """
        Initializes the ARIMA model parameters.

        Args:
            order (tuple): The (p, d, q) parameters for the ARIMA model.
                           p: autoregressive lags, d: differencing, q: moving average.
        """
        self.order = order
        self.model_fit = None
        self._min_window = sum(order) + 1

    def _check_window(self, window, caller):
        if len(window) < self._min_window:
            raise ValueError(
                f"{caller}: window too short — need at least {self._min_window} "
                f"observations for ARIMA{self.order}, got {len(window)}."
            )

    def train(self, train_data):
        """
        Fits the ARIMA model to find optimal coefficients.

        This should be called on the initial 10-20% of your signal. It performs
        the computationally expensive Maximum Likelihood Estimation (MLE) to
        identify the data's pattern.

        Args:
            train_data (np.array or pd.Series): The historical data to train on.
        """
        self._check_window(train_data, "train()")
        with warnings.catch_warnings():
            for w in self._SUPPRESSED:
                warnings.simplefilter("ignore", w)
            model = ARIMA(train_data, order=self.order)
            self.model_fit = model.fit()

    def predict_point(self, window):
        """
        Generates a one-step-ahead forecast using frozen parameters.

        Instead of retraining, this 'applies' the previously learned weights
        to the most recent window of data. This is significantly faster
        than standard ARIMA forecasting.

        Args:
            window (np.array): The most recent observations needed for prediction.

        Returns:
            tuple: (predicted_value, inference_latency)
        """
        if self.model_fit is None:
            raise RuntimeError(
                "Model is not initialised. Call train() before predict_point()."
            )
        self._check_window(window, "predict_point()")

        start_time = time.perf_counter()
        with warnings.catch_warnings():
            for w in self._SUPPRESSED:
                warnings.simplefilter("ignore", w)
            # .apply(window) creates a new Results object using the same
            # coefficients but updated with the latest observations.
            updated_results = self.model_fit.apply(window)
            yhat = float(updated_results.forecast(steps=1)[0])

        latency = time.perf_counter() - start_time
        return yhat, latency
