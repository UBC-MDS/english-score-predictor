import click
import pandas as pd
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
from search_top_models import fit_and_return_top_models
from show_feat_coeff import show_feat_coeff

SCORING = {"RMSE": "neg_root_mean_squared_error", "R squared": "r2"}


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Will print verbose messages.")
@click.option(
    "--train",
    default="data/raw/train_dataset.csv",
    help="Path to the training data.",
)
@click.option(
    "--test",
    default="data/raw/test_dataset.csv",
    help="Path to the test data.",
)
@click.option(
    "--output_dir",
    default="results/models/",
    help="Output directory, default is `results/models/`",
)
def main(
    verbose,
    train,
    test,
    output_dir,
):
    """Runs the analysis of the English Score Tuning."""

    if verbose:
        click.echo("Getting the train and test data...")
    X_train, y_train = get_train_data(train)
    X_test, y_test = get_test_data(test)

    if verbose:
        click.echo("Getting and fitting the preprocessor...")
    preprocessor = get_preprocessor()
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
    ridge_filename = output_dir + "ridge_top_models.csv"
    if verbose:
        click.echo(f"Saving the top models to f{ridge_filename}...")
    ridge_top_models.to_csv(ridge_filename)

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
    lasso_filename = output_dir + "lasso_top_models.csv"
    if verbose:
        click.echo(f"Saving the top models to f{lasso_filename}...")
    lasso_top_models.to_csv(lasso_filename)

    # Discussion Section
    if verbose:
        click.echo("Running the Discussion Section...")
    best_pipe = make_pipeline(
        preprocessor, Ridge(alpha=ridge_search.best_params_["ridge__alpha"])
    )
    best_pipe.fit(X_train, y_train)

    # Get the feature coefficient values
    show_feat_coeff(best_pipe, "ridge", preprocessor)

    # plot actual vs predicted
    PredictionErrorDisplay.from_estimator(
        best_pipe, X_test, y_test, kind="actual_vs_predicted", subsample=None
    ).plot()

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


def get_preprocessor():
    """Returns the preprocessor object from the script"""

    # TODO @farrandi - Change it to actually take it from a file
    numeric_feats = ["age", "Eng_start", "Eng_country_yrs", "Lived_Eng_per"]

    binary_feats = ["psychiatric"]
    target = ["correct"]

    drop_feats = [
        "id",
        "time",
        "Unnamed: 0",
        "tests",
        "elogit",
        "dyslexia",
        "dictionary",
        "already_participated",
        "natlangs",
        "primelangs",
        "Can_region",
        "Ir_region",
        "US_region",
        "UK_region",
        "UK_constituency",
        "gender",
        "type",
        "currcountry",
        "countries",
        "q1",
        "q2",
        "q3",
        "q5",
        "q6",
        "q7",
        "q9_1",
        "q9_4",
        "q10_2",
        "q10_4",
        "q11_3",
        "q11_4",
        "q12_1",
        "q12_2",
        "q12_4",
        "q13_3",
        "q13_4",
        "q14_3",
        "q14_4",
        "q15_1",
        "q15_2",
        "q15_3",
        "q16_3",
        "q16_4",
        "q17_1",
        "q17_3",
        "q17_4",
        "q18_2",
        "q18_3",
        "q18_4",
        "q19_1",
        "q19_2",
        "q19_3",
        "q19_4",
        "q20_1",
        "q20_2",
        "q20_3",
        "q20_4",
        "q21_1",
        "q21_2",
        "q21_3",
        "q21_4",
        "q22_1",
        "q22_2",
        "q22_3",
        "q22_4",
        "q23_3",
        "q23_4",
        "q24_1",
        "q24_2",
        "q24_3",
        "q24_4",
        "q25_1",
        "q25_2",
        "q25_3",
        "q25_4",
        "q26_1",
        "q26_2",
        "q26_3",
        "q26_4",
        "q27_1",
        "q27_2",
        "q27_3",
        "q27_4",
        "q28_1",
        "q28_2",
        "q29_1",
        "q29_2",
        "q29_3",
        "q29_4",
        "q30_1",
        "q30_2",
        "q30_3",
        "q30_4",
        "q31_1",
        "q31_4",
        "q32_5",
        "q32_6",
        "q32_8",
        "q33_4",
        "q33_5",
        "q33_6",
        "q33_7",
        "q34_1",
        "q34_2",
        "q34_3",
        "q34_4",
        "q34_6",
        "q34_8",
        "q35_1",
        "q35_2",
        "q35_4",
        "q35_5",
        "q35_7",
        "q35_8",
    ]

    binary_withNA = [
        "house_Eng",
        "nat_Eng",
        "prime_Eng",
    ]  # ["Ebonics", "house_Eng", "nat_Eng", "prime_Eng"]

    categorical_education = ["education"]  # show 7 major categories and Others
    categorical_feats = ["Eng_little", "speaker_cat"]
    # categorical_countries = ['countries']

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="little"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )

    binary_NA_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, drop="if_binary", dtype=int
        ),
    )

    # Defines the order for Education to be used in the OrdinalEncoder
    education_order = [
        "Graduate Degree",
        "Some Graduate School",
        "Undergraduate Degree (3-5 years higher ed)",
        "Some Undergrad (higher ed)",
        "High School Degree (12-13 years)",
        "Haven't Finished High School (less than 13 years ed)",
        "Didn't Finish High School (less than 13 years ed)",
        "Others",
    ]

    # Define the custom function to map values to 'Other'
    def map_to_other(df):
        return (
            df["education"].apply(lambda x: x if x in education_order else "Others")
        ).to_frame()

    # Create a transformer using FunctionTransformer
    categorical_education_tranformer = make_pipeline(
        FunctionTransformer(map_to_other, feature_names_out="one-to-one"),
        OrdinalEncoder(categories=[education_order]),
    )

    return make_column_transformer(
        ("drop", drop_feats),
        (numeric_transformer, numeric_feats),
        (categorical_transformer, categorical_feats),
        ("passthrough", binary_feats),
        (binary_NA_transformer, binary_withNA),
        (categorical_education_tranformer, categorical_education),
    )


if __name__ == "__main__":
    main()
