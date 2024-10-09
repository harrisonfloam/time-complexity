"""

"""

import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable, Optional, Type, Any
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


MAX_EXP = 16
MAX_EXP_VALUE = 1e10


class TimeComplexity:
    def __init__(self):
        pass
    
    def fit(self, 
            function: Callable, 
            input_size_range: tuple,
            input_range: tuple,
            input_wrapper: Optional[Type[Any]],
            function_kwargs: Optional[dict],
            trials: int=10,
            seed: int=42,
            verbose: bool=False):
        
        #TODO:
        # handling for multiple inputs?
        # 2^k input sizes
        # different types of input
        # np.where handling to constrain input
        
        self.function_name = function.__name__
                
        random.seed(seed)
        
        progress_bar = tqdm(range(input_size_range[0], input_size_range[1]+1), desc="Evaluating Inputs", disable=not verbose)
        times = []
        for input_size in progress_bar:
            for trial in range(trials):
                input = input_wrapper([random.randint(input_range[0], input_range[1]+1) for _ in range(input_size)])
                
                start_time = time.time()
                _ = function(input, **function_kwargs)
                total_time = time.time() - start_time
                
                times.append({"n": input_size, 
                                "trial": trial, 
                                "time": total_time})
        times_df = pd.DataFrame(times)
        
        self.time_stats = times_df.groupby("n").agg(
            avg_time=("time", "mean"),
            std_time=("time", "std")
        ).reset_index()
        
        self._evaluate_complexity()
        
        for complexity, complexity_func in self.complexity_funcs.items():
            i = self.fit_details[self.best_complexity]['intercept'] #BUG: shouldnt this be 0
            c = self.fit_details[self.best_complexity]['coefficient']
            self.time_stats[complexity] = complexity_func(self.time_stats['n'].values, i, c)
            
        if verbose:
            self.report()
            
    def plot(self):
        """"""
        fig, ax = plt.subplots()
        ax.plot(self.time_stats["n"], self.time_stats["avg_time"], label="Actual", color='b')

        ax.fill_between(
            self.time_stats["n"], 
            self.time_stats["avg_time"] - self.time_stats["std_time"], 
            self.time_stats["avg_time"] + self.time_stats["std_time"], 
            color='b', alpha=0.2
        )
        
        for complexity, _ in self.complexity_funcs.items():
            label = f"{complexity}, r2={self.fit_details[complexity]['r2']:.2f}"
            ax.plot(self.time_stats['n'], self.time_stats[complexity], ls='--', label=label)

        max_time = (self.time_stats["avg_time"]).max()
        ax.set_ylim(0, max_time)

        ax.set_xlabel("Input Size (n)")
        ax.set_ylabel("Average Runtime (s)")
        ax.set_title(f"{self.function_name}: {self.best_complexity}")
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()
        ax.grid(True)

        plt.show()
        return fig, ax
                
    def report(self):
        self.complexity_func_strings = {
            # "O(1)": "{i} + {c} * (n)",
            "O(logn)": "{i:.4e} + {c:.4e} * log(n)",
            "O(n)": "{i:.4e} + {c:.4e} * n",
            "O(nlogn)": "{i:.4e} + {c:.4e} * nlog(n)",
            "O(n^2)": "{i:.4e} + {c:.4e} * n^2",
            "O(2^n)": "{i:.4e} + {c:.4e} * 2^n"
        }
        report_data = []
        for complexity, fit_detail in self.fit_details.items():
            intercept = fit_detail["intercept"]
            coefficient = fit_detail["coefficient"]
            r2 = fit_detail["r2"]

            formula = self.complexity_func_strings[complexity].format(i=intercept, c=coefficient)
                
            report_data.append({
                "Complexity": complexity,
                "R^2 Score": f"{r2:.4f}",
                "Formula": formula
            })
        report_df = pd.DataFrame(report_data).sort_values(by='R^2 Score', ascending=False)
        print(report_df.to_string(index=False))
    
    def _evaluate_complexity(self):
        self.complexity_funcs = {
            # "O(1)": lambda n, i, c: i + c * np.ones_like(n),
            "O(logn)": lambda n, i, c: i + c * np.log(n + 1),  # Add 1 for stability
            "O(n)": lambda n, i, c: i + c * n,
            "O(nlogn)": lambda n, i, c: i + c * n * np.log(n + 1),
            "O(n^2)": lambda n, i, c: i + c * n ** 2,
            "O(2^n)": lambda n, i, c: i + c * np.where(n <= MAX_EXP, np.power(2, n.astype(float)), MAX_EXP_VALUE)
        }
        
        input_sizes = self.time_stats["n"].values.reshape(-1, 1)
        avg_times = self.time_stats["avg_time"].values
        
        best_fit = None
        best_score = -np.inf
        best_complexity = None
        fit_details = {}
        
        for complexity, complexity_func in self.complexity_funcs.items():
            transformed_n = complexity_func(input_sizes, 0, 1).reshape(-1, 1)
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(transformed_n, avg_times)
            predicted_times = model.predict(transformed_n)
            r2 = r2_score(avg_times, predicted_times)
            
            intercept = model.intercept_
            coef = model.coef_[0]
            fit_details[complexity] = {
                "r2": r2,
                "intercept": intercept,
                "coefficient": coef
            }
            
            # Select best fit
            if r2 > best_score:
                best_score = r2
                best_complexity = complexity
                best_fit = model

        self.best_fit = best_fit
        self.best_complexity = best_complexity
        self.fit_details = fit_details
    
    
def time_complexity(function: Callable, 
                    input_size_range: tuple,
                    input_range: tuple,
                    input_wrapper: Optional[Type[Any]],
                    function_kwargs: Optional[dict],
                    trials: int=10,
                    seed: int=42,
                    plot: bool=True,
                    verbose: bool=False):
    pass

#TODO: just a function that returns a dataframe? with an option to plot?