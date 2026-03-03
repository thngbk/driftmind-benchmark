import time
import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA


class FrozenARIMABaseline:
    """
    A stateful, low-latency ARIMA implementation for streaming inference.

    STRATEGY:
    This model employs a 'Frozen Parameter' approach. It performs expensive
    Maximum Likelihood Estimation (MLE) only once during the training phase.
    During inference, it updates the 'Kalman Filter' state of the model with
    new observations using .append(), but refuses to recalculate the
    underlying coefficients (refit=False).

    ANGLE IN EXPERIMENT:
    This serves as the 'Efficient Memory' baseline. It is faster than an
    adaptive model because it never retrains, but it is more 'aware' of the
    streaming history than a simple static window approach.
    """

    _SUPPRESSED = (ConvergenceWarning, UserWarning)

    def __init__(self, order=(5, 1, 0)):
        """
        Initializes the ARIMA model configuration.

        Args:
            order (tuple): The (p, d, q) stochastic process parameters.
                           (5, 1, 0) implies a 5th-order Autoregressive
                           process on a first-differenced signal.
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

    def train(self, train_series):
        """
        Initializes the model parameters (coefficients).

        This method should be executed on a representative 'warm-up' period
        of the data. The coefficients discovered here will be 'frozen'
        for the remainder of the experiment.

        Args:
            train_series (array-like): Historical data used to find the
                                       optimal model weights.
        """
        self._check_window(train_series, "train()")
        with warnings.catch_warnings():
            for w in self._SUPPRESSED:
                warnings.simplefilter("ignore", w)
            model = ARIMA(train_series, order=self.order)
            self.model_fit = model.fit()

    def predict_point(self, new_obs=None):
        """
        Executes a 1-step ahead forecast and updates internal state.

        This method handles the transition from Time T to Time T+1.
        1. It generates a forecast based on all data seen up to now.
        2. It accepts the actual observed value (new_obs) to update the
           model's internal buffer for the next prediction.

        Args:
            new_obs (float, optional): The actual value observed after
                                       the last prediction.

        Returns:
            tuple: (yhat, latency)
                yhat (float): The predicted value for the next time step.
                latency (float): The time in seconds taken to forecast
                                 and update state.
        """
        if self.model_fit is None:
            raise RuntimeError("Call train() first.")

        start = time.perf_counter()

        with warnings.catch_warnings():
            for w in self._SUPPRESSED:
                warnings.simplefilter("ignore", w)
            # Generate a 1-step ahead forecast from the current 'frozen' state
            yhat = float(self.model_fit.forecast(steps=1)[0])
            # State Update: Logic for 'Memory' without 'Learning'
            if new_obs is not None:
                # .append() extends the model's history.
                # refit=False ensures we do NOT trigger a new MLE optimization,
                # keeping inference latency consistent and low.
                self.model_fit = self.model_fit.append([new_obs], refit=False)

        latency = time.perf_counter() - start
        return yhat, latency
