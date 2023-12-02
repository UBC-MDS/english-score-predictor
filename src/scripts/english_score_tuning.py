import click
import pandas as pd
import pickle
import sys
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import train_test_split

sys.path.append("src")
from helper.search_top_models import fit_and_return_top_models
from helper.show_feat_coeff import show_feat_coeff

SCORING = {"RMSE": "neg_root_mean_squared_error", "R squared": "r2"}


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Will print verbose messages.")
@click.option(
    "--train",
    default="data/raw/train_data.csv",
    help="Path to the training data.",
)
@click.option(
    "--test",
    default="data/raw/test_data.csv",
    help="Path to the test data.",
)
@click.option(
    "--output_dir",
    default="results/",
    help="Output directory, default is `results/`",
)
@click.option(
    "--preprocessor_path",
    default="results/models/preprocessor/preprocessor.pkl",
    help="Path to the preprocessor object (default: results/models/preprocessor/preprocessor.pkl)",
)
def main(verbose, train, test, output_dir, preprocessor_path):
    """Runs the analysis of the English Score Tuning."""

    if verbose:
        click.echo("Getting the train and test data...")
    X_train, y_train = get_train_data(train)
    X_test, y_test = get_test_data(test)

    if verbose:
        click.echo("Getting and fitting the preprocessor...")
    preprocessor = get_preprocessor(preprocessor_path)
    preprocessor.fit(X_train)

    # Ridge Regression
    if verbose:
        click.echo("Running the Ridge Regression...")
    ridge_pipe = make_pipeline(preprocessor, Ridge())

    param_grid = {
        "ridge__alpha": loguniform(1e-3, 1e3),
    }

    ridge_search = RandomizedSearchCV(
        ridge_pipe,
        param_distributions=param_grid,
        n_jobs=-1,
        n_iter=30,
        cv=10,
        return_train_score=True,
        random_state=123,
        scoring=SCORING,
        refit="RMSE",
    )

    ridge_top_models = fit_and_return_top_models(
        ridge_search,
        5,
        X_train,
        y_train,
        [
            "param_ridge__alpha",
            "mean_train_R squared",
            "mean_test_R squared",
        ],
        scoring="RMSE",
    )

    # Saves the top models to a csv file
    ridge_filename = output_dir + "tables/ridge_top_models.csv"
    if verbose:
        click.echo(f"Saving the top models to {ridge_filename}")
    ridge_top_models.to_csv(ridge_filename)

    # Saves the best model to a pickle file
    ridge_best_model = ridge_search.best_estimator_
    ridge_best_model_filename = output_dir + "models/ridge_best_model.pkl"
    if verbose:
        click.echo(f"Saving the best model to {ridge_best_model_filename}")
    with open(ridge_best_model_filename, "wb") as f:
        pickle.dump(ridge_best_model, f)

    # Lasso Regression
    if verbose:
        click.echo("Running the Lasso Regression...")
    lasso_dist = {"lasso__alpha": loguniform(1e-3, 1e3)}

    lasso_pipe = make_pipeline(preprocessor, Lasso())
    lasso_search = RandomizedSearchCV(
        lasso_pipe,
        param_distributions=lasso_dist,
        n_jobs=-1,
        n_iter=30,
        cv=10,
        return_train_score=True,
        random_state=123,
        scoring=SCORING,
        refit="RMSE",
    )

    lasso_top_models = fit_and_return_top_models(
        lasso_search,
        5,
        X_train,
        y_train,
        [
            "param_lasso__alpha",
            "mean_train_R squared",
            "mean_test_R squared",
        ],
        scoring="RMSE",
    )

    # Saves the top models to a csv file
    lasso_filename = output_dir + "tables/lasso_top_models.csv"
    if verbose:
        click.echo(f"Saving the top models to {lasso_filename}")
    lasso_top_models.to_csv(lasso_filename)

    # Saves the best model to a pickle file
    lasso_best_model = lasso_search.best_estimator_
    lasso_best_model_filename = output_dir + "models/lasso_best_model.pkl"
    if verbose:
        click.echo(f"Saving the best model to {lasso_best_model_filename}")
    with open(lasso_best_model_filename, "wb") as f:
        pickle.dump(lasso_best_model, f)

    if verbose:
        click.echo("Done!")


def get_train_data(train):
    """Returns the training data as a tuple of X_train and y_train"""

    train_df = pd.read_csv(train, sep=",", on_bad_lines="skip", low_memory=False)
    X_train = train_df.drop(columns="correct")
    y_train = train_df["correct"]

    return X_train, y_train


def get_test_data(test):
    """Returns the test data as a tuple of X_test and y_test"""

    test_df = pd.read_csv(test, sep=",", on_bad_lines="skip", low_memory=False)
    X_test = test_df.drop(columns="correct")
    y_test = test_df["correct"]

    return X_test, y_test


def get_preprocessor(path):
    """Returns the preprocessor object from the path (pickle file)"""
    with open(path, "rb") as f:
        preprocessor = pickle.load(f)

    return preprocessor


def map_to_other(df):
    """helper function to map categories to others, used in the preprocessor"""
    categories_list = [
        "Graduate Degree",
        "Some Graduate School",
        "Undergraduate Degree (3-5 years higher ed)",
        "Some Undergrad (higher ed)",
        "High School Degree (12-13 years)",
        "Haven't Finished High School (less than 13 years ed)",
        "Didn't Finish High School (less than 13 years ed)",
        "Others",
    ]

    return (
        df["education"].apply(lambda x: x if x in categories_list else "Others")
    ).to_frame()


if __name__ == "__main__":
    main()
