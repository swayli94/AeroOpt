'''
This is an interface for surrogate models in aeroopt.

Classic surrogate model packages:

- `SMT`: Surrogate Modeling Toolbox
    - website: https://smt.readthedocs.io/en/latest/index.html
    - radial basis function (RBF) surrogate model;
    - kriging model that uses the partial least squares (PLS) method;
    - Multi-Fidelity Co-Kriging (MFCK);
'''

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict

from aeroopt.core.problem import Problem


class SurrogateModel(ABC):
    '''
    Base class for surrogate models.
    
    Parameters:
    -----------
    problem: Problem
        Problem of the Database.
    model_name: str
        Name of the surrogate model.
    train_on_scaled_data: bool
        If True, train the surrogate model on the scaled input/output data.
        If False, train the surrogate model on the original input/output data.
        
    Attributes:
    -----------
    model: Any
        Surrogate model object.
        The type of the model is determined by the package (e.g., `SMT`).
    size: int
        Size of the training data.
        The number of training data is equal to the size of the database.
    '''
    def __init__(self, problem: Problem, model_name: str = 'default',
                train_on_scaled_data: bool = True):
        
        self.problem = problem
        self.model_name = model_name
        self.train_on_scaled_data = train_on_scaled_data
        
        self._model : Any = None
        self._size : int = 0
    
    @property
    def model(self) -> Any:
        '''
        Surrogate model object.
        '''
        return self._model
    
    @property
    def n_input(self) -> int:
        '''
        Number of input variables.
        '''
        return self.problem.n_input
    
    @property
    def n_output(self) -> int:
        '''
        Number of output variables.
        '''
        return self.problem.n_output
    
    @property
    def size(self) -> int:
        '''
        Size of the training data.
        '''
        return self._size
        
    @abstractmethod
    def train(self, xs: np.ndarray, ys: np.ndarray) -> None:
        '''
        Train the surrogate model using the input/output data.
        
        Parameters:
        -----------
        xs: np.ndarray [n, n_input]
            Original input data.
        ys: np.ndarray [n, n_output]
            Original output data.
        '''
        pass
    
    @abstractmethod
    def predict(self, xs: np.ndarray) -> np.ndarray:
        '''
        Predict the output using the input.
        
        Parameters:
        -----------
        xs: np.ndarray [n, n_input]
            Original input data.
            
        Returns:
        --------
        ys: np.ndarray [n, n_output]
            Predicted original output data.
        '''
        pass

    @abstractmethod
    def full_predict(self, xs: np.ndarray) -> Dict[str, Any]:
        '''
        Predict the output using the input,
        including additional information such as the confidence interval, etc.
        
        Parameters:
        -----------
        xs: np.ndarray [n, n_input]
            Original input data.
            
        Returns:
        --------
        result: Dict[str, Any]
            Result dictionary containing all information.
            The optional keys of the dictionary are:
            - 'ys': np.ndarray [n, n_output]
                Predicted original output data (mean value).
            - 'epistemic_variance': np.ndarray [n, n_output]
                Epistemic variance of the prediction.
            - 'aleatoric_variance': np.ndarray [n, n_output]
                Aleatoric variance of the prediction.
        '''
        pass
    
    @abstractmethod
    def predict_for_adaptive_sampling(self, xs: np.ndarray) -> np.ndarray:
        '''
        Predict the objective values for adaptive sampling.
        
        Parameters:
        -----------
        xs: np.ndarray [n, n_input]
            Input data.
            
        Returns:
        --------
        objectives: np.ndarray [n, n_output]
            Predicted objective values for adaptive sampling, e.g.,
            - the expected improvement (EI)
            - upper/lower confidence bound (UCB/LCB)
            - space-filling criterion (SFC)
            - etc.
        '''
        pass
    
    @abstractmethod
    def evaluate_performance(self, xs: np.ndarray, ys_actual: np.ndarray) -> Dict[str, Any]:
        '''
        Evaluate the performance of the surrogate model by comparing the prediction
        and actual values of the individuals.
        
        Parameters:
        -----------
        xs: np.ndarray [n, n_input]
            Input data.
        ys_actual: np.ndarray [n, n_output]
            Actual output data.
            
        Returns:
        --------
        result: Dict[str, Any]
            Result dictionary containing all information.
            The optional keys of the dictionary are:
            - 'metric': float
                Performance metric of the surrogate model.
            - 'RMSE': np.ndarray [n_output]
                Root Mean Square Error of the prediction.
            - 'MAE': np.ndarray [n_output]
                Mean Absolute Error of the prediction.    
            - 'NLL': np.ndarray [n_output]
                Negative Log Likelihood of the prediction.
        '''
        pass


class Kriging(SurrogateModel):
    '''
    Kriging surrogate model from `SMT` package.
    
    Parameters:
    -----------
    problem: Problem
        Problem of the Database.
    model_name: str
        Name of the surrogate model.
    train_on_scaled_data: bool
        If True, train the surrogate model on the scaled input/output data.
    **kwargs
        Forwarded to `smt.surrogate_models.KPLS`. By default `print_global`
        is False so SMT does not print training/prediction banners to stdout.
    '''
    def __init__(self, problem: Problem, model_name: str = 'Kriging',
                train_on_scaled_data: bool = True, **kwargs):
        
        from smt.surrogate_models import KPLS
        
        self.problem = problem
        self.model_name = model_name
        self.train_on_scaled_data = train_on_scaled_data

        kpls_kwargs = {'print_global': False}
        kpls_kwargs.update(kwargs)
        self._model = [KPLS(**kpls_kwargs) for _ in range(problem.n_output)]
        self._size : int = 0
                
    @property
    def output_span(self) -> np.ndarray:
        '''
        Span of the output variables, used for scaling the epistemic standard deviation.
        '''
        span = self.problem.data_settings.output_upp - self.problem.data_settings.output_low
        return span
    
    @property
    def output_type(self) -> list[int]:
        '''
        Type of the output variables.
        '''
        return self.problem.problem_settings.output_type
    
    def _get_sampling_criteria(self, ys: np.ndarray, epistemic_std: np.ndarray) -> np.ndarray:
        '''
        Get the criteria for adaptive sampling.
        
        Parameters:
        -----------
        ys: np.ndarray [n, n_output]
            Output data.
        epistemic_std: np.ndarray [n, n_output]
            Epistemic standard deviation of the output.
            
        Returns:
        --------
        criteria: np.ndarray [n, n_output]
            Criteria for adaptive sampling.
        '''
        criteria = np.zeros_like(ys)
        
        for i in range(self.n_output):
            
            if self.output_type[i] == 1:
                # Maximization: upper confidence bound (UCB), aligns with
                # `get_unified_objectives` (larger criterion is better).
                criteria[:, i] = ys[:, i] + epistemic_std[:, i]
                
            elif self.output_type[i] == -1:
                # Minimization: lower confidence bound (LCB); after negation in
                # unified objectives, smaller LCB yields larger unified value.
                criteria[:, i] = ys[:, i] - epistemic_std[:, i]
                
            else:
                # Additional output: prefer regions with large epistemic uncertainty
                criteria[:, i] = epistemic_std[:, i]
        
        return criteria
    
    def train(self, xs: np.ndarray, ys: np.ndarray) -> None:
        '''
        Train the surrogate model using the input/output data.
        '''
        if self.train_on_scaled_data:
            xt = self.problem.scale_x(xs)
            yt = self.problem.scale_y(ys)
        else:
            xt = xs
            yt = ys
        
        for i in range(self.n_output):
            self._model[i].set_training_values(xt, yt[:, i])
            self._model[i].train()
        self._size = xs.shape[0]
        
    def predict(self, xs: np.ndarray) -> np.ndarray:
        '''
        Predict the output using the input.
        '''
        ys = np.zeros((xs.shape[0], self.n_output))
        
        if self.train_on_scaled_data:
            
            xt = self.problem.scale_x(xs)
            for i in range(self.n_output):
                pred = np.asarray(self._model[i].predict_values(xt), dtype=float)
                ys[:, i] = pred.reshape(-1)
            ys = self.problem.scale_y(ys, reverse=True)
            
        else:
            
            for i in range(self.n_output):
                pred = np.asarray(self._model[i].predict_values(xs), dtype=float)
                ys[:, i] = pred.reshape(-1)
        
        return ys
    
    def full_predict(self, xs: np.ndarray) -> Dict[str, Any]:
        '''
        Predict the output using the input,
        including additional information such as the confidence interval, etc.
        '''
        ys = np.zeros((xs.shape[0], self.n_output))
        epistemic_variance = np.zeros((xs.shape[0], self.n_output))
        
        if self.train_on_scaled_data:
            
            xt = self.problem.scale_x(xs)
            for i in range(self.n_output):
                pred = np.asarray(self._model[i].predict_values(xt), dtype=float)
                var = np.asarray(self._model[i].predict_variances(xt), dtype=float)
                ys[:, i] = pred.reshape(-1)
                epistemic_variance[:, i] = var.reshape(-1)
            
            ys = self.problem.scale_y(ys, reverse=True)
            output_span = self.output_span
            epistemic_variance = epistemic_variance * output_span[None, :] ** 2
            
        else:
            
            for i in range(self.n_output):
                pred = np.asarray(self._model[i].predict_values(xs), dtype=float)
                var = np.asarray(self._model[i].predict_variances(xs), dtype=float)
                ys[:, i] = pred.reshape(-1)
                epistemic_variance[:, i] = var.reshape(-1)
            
        return {"ys": ys, "epistemic_variance": epistemic_variance}
    
    def evaluate_performance(self, xs: np.ndarray, ys_actual: np.ndarray) -> Dict[str, Any]:
        '''
        Evaluate the performance of the surrogate model by comparing the prediction
        and actual values of the individuals.
        
        Using scaled data for evaluation when `train_on_scaled_data` is True.
        '''
        if self.train_on_scaled_data:
            xs = self.problem.scale_x(xs)
            ys_actual = self.problem.scale_y(ys_actual)

        ys_pred = np.zeros((xs.shape[0], self.n_output))
        for i in range(self.n_output):
            pred = np.asarray(self._model[i].predict_values(xs), dtype=float)
            ys_pred[:, i] = pred.reshape(-1)

        rmse = np.sqrt(np.mean((ys_pred - ys_actual) ** 2, axis=0))
        mae = np.mean(np.abs(ys_pred - ys_actual), axis=0)

        return {"RMSE": rmse, "MAE": mae}
    
    def predict_for_adaptive_sampling(self, xs: np.ndarray) -> np.ndarray:
        '''
        Predict the objective values for adaptive sampling.
        '''
        result = self.full_predict(xs)
        criteria = self._get_sampling_criteria(
            ys=result["ys"],
            epistemic_std=np.sqrt(result["epistemic_variance"]),
        )
        return criteria
    