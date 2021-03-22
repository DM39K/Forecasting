import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.stats.stattools import durbin_watson

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import matplotlib.pyplot as plt

metrics = { 
           'RMSE': lambda real, predictions: np.sqrt(sum((predictions-real.values)**2)/len(real)),
           'RSS' : lambda real, predictions: sum((predictions-real.values)**2), 
           'MSE' : mean_squared_error, 
           'MAE' : mean_absolute_error,
#            'MAPE': mean_absolute_percentage_error, 
           'R2'  : r2_score 
          }

def calculate_metrics(predictions, real, metrics=metrics):
    
    result = {}
    
    for name, function in metrics.items():
        
        calculated_metric = function(real, predictions)
        result.update({name:calculated_metric})
        
    return result

def volatile_models_metrics(input_model):
    return { 'AIC': input_model.aic,
             'BIC' : input_model.bic, 
             'R-squared' : input_model.rsquared,
             'DW' : durbin_watson(input_model.resid.dropna())
           }

def simple_plot(ts1, ts2, figsize=(10,8), legend=['A','B'], title='Time Series'):
    ts1.plot(figsize=figsize)
    ts2.plot(figsize=figsize)
    plt.legend(legend)
    plt.title(title)
    plt.show()
    
def predict(coef, history, steps):
    y = []
    df = [x for x in history]
    
    for k in range(steps):
        temp = 0.
        for i in range(1, len(coef)+1):
            temp += coef[i-1] * df[-i]
        y.append(temp)
        df.append(temp)
        
    return np.array(y)

def stationarity_test(ts, stat_test):
    
    result = stat_test(ts)
    test_name = stat_test.__name__
    
    test_result = {
        test_name + "_statistics": result[0],
        test_name + "_p_value": result[1],
    }

    p = 4 if test_name == 'adfuller' else 3 if test_name == 'kpss' else None
    
    for key,value in result[p].items():
        test_result[test_name+'_critical_value (%s)'%key] = value
    
    return test_result