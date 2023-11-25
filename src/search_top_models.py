from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import pandas as pd

scoring_dict = {
    "r2": "r2",
    "sklearn MAPE": "neg_mean_absolute_percentage_error",
    "neg_root_mean_square_error": "neg_root_mean_squared_error",
    "neg_mean_squared_error": "neg_mean_squared_error",
}


def fit_and_return_top_models(
    search,
    N,
    X_train,
    y_train,
    additional_columns=[],
    return_train_score=True,
    scoring="score",
):
    """
    Fits a SearchCV object and returns the top N models as a DataFrame.

    Parameters
    ----------
    search : RandomizedSearchCV or GridSearchCV
        A RandomizedSearchCV or GridSearchCV object.
    N : int
        The number of top models to return.
    X_train : pandas.DataFrame
        The training data.
    y_train : pandas.Series
        The training labels.
    additional_columns : list, optional
        A list of additional columns to display. Defaults to [].
    return_train_score : bool, optional
        Whether to return the training scores. Defaults to True.
    scoring : str, optional
        The scoring metric to use. Defaults to "score".
        Can be any other name if a dictionary was passed to scoring in RandomizedSearchCV/GridSearchCV.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the top N models.

    Examples
    --------
    >>> fit_and_return_top_models(ridge_search, 5, X_train, y_train, "RMSE", True, "ridge__alpha")

    """
    # fit the search
    search.return_train_score = return_train_score
    search.fit(X_train, y_train)

    # Set the columns to display
    columns = [
        "rank_test_" + scoring,
        "mean_test_" + scoring,
        "mean_fit_time",
    ]

    if return_train_score:
        columns.append("mean_train_" + scoring)

    # add the columns to display
    columns.extend(additional_columns)

    # return the top N models as a DataFrame
    return (
        pd.DataFrame(search.cv_results_)[columns]
        .set_index("rank_test_" + scoring)
        .sort_index()
        .head(N)
        .T
    )
