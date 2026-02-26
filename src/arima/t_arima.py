import warnings
import time
from statsmodels.tsa.arima.model import ARIMA

class TriggeredARIMABaseline:
    """
    A reactive ARIMA implementation designed for triggered adaptation.
    
    STRATEGY:
    This class treats the model as a two-speed system:
    1. Fast Mode (predict_point): Uses fixed parameters to update the internal 
       Kalman Filter state and generate forecasts.
    2. Slow Mode (train): Performs a full Maximum Likelihood Estimation (MLE) 
       to find new optimal coefficients when a trigger is fired.
    
    ANGLE IN EXPERIMENT:
    This represents the 'Smart Hybrid' baseline. It mimics the behavior of 
    advanced drift-detection systems by remaining computationally efficient 
    during stability and only incurring 'training debt' when performance decays.
    """

    def __init__(self, order=(5, 1, 0)):
        """
        Initializes the ARIMA configuration.

        Args:
            order (tuple): The (p, d, q) parameters defining the model structure.
        """
        self.order = order
        self.model_fit = None

    def train(self, window):
        """
        Forces a comprehensive re-optimization of model parameters.

        This method should be called when an external monitor detects drift. 
        It uses 'warm-starting'—passing previous parameters to the optimizer—
        to potentially speed up convergence on the new data window.

        Args:
            window (array-like): The historical data window used for refitting.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(window, order=self.order)
            
            # Optimization: Warm-start using current params if they exist
            start_params = self.model_fit.params if self.model_fit else None
            self.model_fit = model.fit(start_params=start_params)

    def predict_point(self, window):
        """
        Performs high-speed inference by updating only the model state.

        This uses the .apply() method to synchronize the model's 'memory' 
        with the current window of data without touching the coefficients. 
        This ensures the forecast is based on the most recent values while 
        maintaining extremely low latency.

        Args:
            window (array-like): The context window required by the model order.

        Returns:
            tuple: (yhat, latency)
                yhat (float): The 1-step ahead prediction.
                latency (float): Time in seconds for the state update and forecast.
        """
        if self.model_fit is None:
            # Automatic initialization if called before training
            self.train(window)
        
        start_time = time.perf_counter()
        
        # .apply updates the filtered state for the window (math without learning)
        updated_model = self.model_fit.apply(window)
        yhat = float(updated_model.forecast(steps=1)[0])
        
        latency = time.perf_counter() - start_time
        return yhat, latency