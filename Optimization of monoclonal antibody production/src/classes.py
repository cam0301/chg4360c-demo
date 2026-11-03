# =======================================================================
# CLASSES
# Define custom classes in this file.
# =======================================================================
import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class MabProductionPolyModel:
    def __init__(self):
        self.model_gscv = None
    def optimal_hyperparam(self, x_train, y_train, groups):
        # Create polynomial regression pipeline
        model = make_pipeline(StandardScaler(),PolynomialFeatures(include_bias=False),LinearRegression())

        #Define polynomial degree range
        param_grid = dict(polynomialfeatures__degree=[i for i in range(1, 4)])

        # Create the model using GridSearchCV
        # Keep batches separate
        gkf = GroupKFold(n_splits=3)

        self.model_gscv = GridSearchCV(model, param_grid=param_grid,
                                  cv=gkf, scoring="neg_root_mean_squared_error", return_train_score=True)

        # Fit model using training data
        self.model_gscv.fit(x_train, y_train, groups=groups)

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

class MabProductionKNNModel:
    def __init__(self):
        self.model_gscv = None

    def optimal_hyperparam(self, x_train, y_train, groups):
        # Create KNN regression pipeline
        model = make_pipeline(StandardScaler(), KNeighborsRegressor())

        # Define the "Grid" that we want to "Search" when performing "CV" in GridSearchCV
        param_grid = dict(
            kneighborsregressor__n_neighbors=[i for i in range(2, 6)],
            kneighborsregressor__weights=["uniform"]
        )

        # Create the model using GridSearchCV
        # Keep batches separate
        gkf = GroupKFold(n_splits=3)

        self.model_gscv = GridSearchCV(model, param_grid=param_grid, cv=gkf, scoring="neg_root_mean_squared_error",
                                  return_train_score=True)

        # Train model
        self.model_gscv.fit(x_train, y_train, groups=groups)

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