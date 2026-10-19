# =======================================================================
# CLASSES
# Define custom classes in this file.
# =======================================================================
import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures



class CellGrowthModel:
    def __init__(self):
        self.model_gscv = None

    def optimal_hyperparam(self, x_train, y_train):
        # Create polynomial regression pipeline
        model = make_pipeline(StandardScaler(),PolynomialFeatures(include_bias=False),LinearRegression())

        #Define polynomial degree range
        param_grid = dict(polynomialfeatures__degree=[i for i in range(1, 6)])

        # Create the model using GridSearchCV
        self.model_gscv = GridSearchCV(model, param_grid=param_grid,
                                  cv=4, scoring="neg_root_mean_squared_error", return_train_score=True)

        # Fit model using training data
        self.model_gscv.fit(x_train, y_train)

        # Place the results in a DataFrame
        df = pd.DataFrame(self.model_gscv.cv_results_)

        # Sort the DataFrame by rank_test_score
        df = df.sort_values(by=['rank_test_score'], ascending=True)

        # Define the columns of interest
        columns_view = ["rank_test_score", "param_polynomialfeatures__degree",
                        "mean_test_score", "std_test_score",
                        "mean_train_score", "std_train_score"]

        # Print results
        print("\nPolynomial Regression Cross-Validation Results:\n")
        print(df.loc[:, columns_view].to_string())

        # Export DataFrame
        df.loc[:, columns_view].to_csv(
            os.path.join("tables", f"table_01_polynomial_cv_results.csv"),
            index=False)

    def evaluate_model(self, x_test, y_test):
        test_score = self.model_gscv.score(x_test, y_test)
        return test_score

    def predict_targets(self, x_test):
        predictions = self.model_gscv.predict(x_test)
        return predictions

class KNNComparisonModel:
    def __init__(self):
        self.model_gscv = None

    def optimal_hyperparam(self, x_train, y_train):
        # Create KNN regression pipeline
        model = make_pipeline(StandardScaler(), KNeighborsRegressor())

        # Define the "Grid" that we want to "Search" when performing "CV" in GridSearchCV
        param_grid = dict(
            kneighborsregressor__n_neighbors=[i for i in range(1, 81)],  # Try every value of n_neighbors from 1 to 80
            kneighborsregressor__weights=["uniform", "distance"],  # Try both weight algorithms
        )

        # Create the model using GridSearchCV
        self.model_gscv = GridSearchCV(model, param_grid=param_grid, cv=4, scoring="neg_root_mean_squared_error",
                                  return_train_score=True)

        self.model_gscv.fit(x_train, y_train)

        df = pd.DataFrame(self.model_gscv.cv_results_)

        # Sort the DataFrame by rank_test_score (i.e., best to worst)
        df = df.sort_values(by=['rank_test_score'], ascending=True)

        # Define the columns of interest with the relevant hyperparameters
        columns_view = [
            "rank_test_score",
            "param_kneighborsregressor__n_neighbors",
            "param_kneighborsregressor__weights",
            "mean_test_score", "std_test_score",
            "mean_train_score", "std_train_score"
        ]
        # Print results
        print("\nKNN Cross-Validation Results:\n")
        print(df.loc[:, columns_view].to_string())

        # Export DataFrame
        df.loc[:, columns_view].to_csv(
            os.path.join("tables", f"table_02_knn_cv_results.csv"),
            index=False)

    def evaluate_model(self, x_test, y_test):
        test_score = self.model_gscv.score(x_test, y_test)
        return test_score

    def predict_targets(self, x_test):
        predictions = self.model_gscv.predict(x_test)
        return predictions