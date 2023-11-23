import pandas as pd


def show_feat_coeff(pipe_obj, model_name, X):
    """
    Show the estimated feature coefficients for a learned regression model. 

    Creates a new DataFrame with one column, whose index is the column names of the input 
    array of examples, listing the estimated feature coefficients for the corresponding
    features in the input array.

    Parameters:
    ----------
    pipe_obj : Pipeline
        The fitted input pipeline of transformers with a regression model estimator. Pipeline must be fitted to 
        get coefficient values. 
    model_name : str
        The name of the regression model as per the Pipeline object named steps.
    X : array-like shape (n_samples, n_features)
        The input array-like shape containing the training data on which the input Pipeline object was fit.
        The n_features dimension of the array-like shape will indicate the number of coefficient values to 
        be returned corresponding to the column names for the n_features.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with one column:
        - 'Coefficients': Lists the estimated feature coefficients for the input regression model.
        whose index is the column names of the input array of test samples (X)
        
    Examples:
    --------
    >>> import pandas as pd
    >>> from sklearn.pipeline import Pipeline 
    >>> from sklearn.linear_model import Lasso, Ridge, LinearRegression
    # Replace StandardScaler with desired preprocessing step(s)
    # Replace Lasso model with desired regression model (e.g. Ridge, LinearRegression)
    >>> pipe = Pipeline([('stdsclr', StandardScaler()), ('lasso', Lasso())]) 
    >>> pipe.fit(X_train, y_train)
    >>> results_df = show_feat_coeff(pipe, 'lasso', X_train)
    >>> results_df
    
    Notes:
    -----
    This function uses the pandas library to return result as DataFrame.

    """
    # Access .coef_ attribute of regression model 
    coef_vals = pipe_obj.named_steps[model_name].coef_
    coef_id = X.columns
    # Report coefficient values and corresponding feature names in a DataFrame
    result = pd.DataFrame(
        data=coef_vals, index=coef_id, columns=["Coefficients"]
        ).sort_values(by="Coefficients", ascending=False)
    return result