import click
import pandas as pd
import pickle
import sys
import dataframe_image as dfi
from show_feat_coeff import show_feat_coeff
from sklearn.metrics import PredictionErrorDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

sys.path.append('src')

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
    "--plot_to",
    type=str,
    default="results/figures/",
    help="Path to where all plots should be written ex: results/figures/",
)
@click.option(
    "--tables_to",
    type=str,
    default="results/tables/",
    help="Path to where tables should be written ex: results/tables/",
)
@click.option(
    "--preprocessor_path",
    default="results/models/preprocessor/preprocessor.pkl",
    help="Path to the preprocessor object (default: results/models/preprocessor/preprocessor.pkl)",
)
@click.option(
    "--pipeline_path",
    default="results/models/preprocessor/preprocessor.pkl",
    help="Path to the preprocessor object (default: results/models/preprocessor/preprocessor.pkl)",
)

def main(verbose, train, test, plot_to, tables_to, preprocessor_path, pipeline_path):
    '''
        Main function for discussion and results of final model. 
        Includes reporting test score, showing feature coefficients, and plotting actual vs. predicted values

        Args:
            verbose (str): flag to print verbose messages.
            train (str): Path to the train data CSV file.
            test (str): Path to the test data CSV file.
            plot_to (str): Path to save the generated plots.
            tables_to (str): Path to save the tables.
            preprocessor_path (str): Path to preprocessor object.
            pipeline_path (str): Path to pipeline object.
    Returns:
        None
    '''

    if verbose:
        click.echo("Getting the train and test data...")
    # Get train data
    train_df = pd.read_csv(train, sep=",", on_bad_lines="skip", low_memory=False)
    X_train = train_df.drop(columns="correct")
    y_train = train_df["correct"]
    # Get test data
    test_df = pd.read_csv(test, sep=",", on_bad_lines="skip", low_memory=False)
    X_test = test_df.drop(columns="correct")
    y_test = test_df["correct"]

    if verbose:
        click.echo("Getting and fitting the preprocessor...")
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    preprocessor.fit(X_train, y_train)

    if verbose:
        click.echo("Getting and fitting the pipeline...")
    with open(pipeline_path, "rb") as f:
        best_pipeline = pickle.load(f)
    best_pipeline.fit(X_train, y_train)

    if verbose:
        click.echo("Getting Best Test Score...")
    # Get the test score of the best model
    score = pd.DataFrame(
        {
            "r2" : best_pipeline.score(X_test, y_test),
            "rmse" : (mean_squared_error(X_test, best_pipeline.predict(X_test)) ** 1/2),
            "mape" : mean_absolute_percentage_error
        }
    )
    score.data.to_csv(tables_to + "test-score.csv")

    if verbose:
        click.echo("Running the feature coefficient section...")
    # Get the feature coefficient values
    sfc = show_feat_coeff(best_pipeline, "ridge", preprocessor)
    dfi.export(sfc, plot_to + "feat-coefs.png")
    sfc.data.to_csv(tables_to + "feat-coefs.csv")

    if verbose:
        click.echo("Plotting the actual vs predicted...")
    # Plot actual vs predicted
    ped = PredictionErrorDisplay.from_estimator(
        best_pipeline, X_test, y_test, kind="actual_vs_predicted", subsample=None
    ).plot()
    dfi.export(ped, plot_to + "act-vs-pred.png")

    if verbose:
        click.echo("Done!")
            
    if __name__ == "__main__":
        main()