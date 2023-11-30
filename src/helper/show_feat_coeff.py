import pandas as pd


def show_feat_coeff(pipe_obj, model_name, preprocessor_obj):
    """
    Show the estimated feature coefficients for a learned regression model. 

    Creates a new DataFrame with one column, whose index is the column names of the input 
    data frame of examples, listing the estimated feature coefficients for the corresponding
    features in the input data frame.

    Parameters:
    ----------
    pipe_obj : Pipeline
        The fitted input pipeline of transformers with a regression model estimator. Pipeline must be fitted to 
        get coefficient values. 
    model_name : str
        The name of the regression model as per the Pipeline object named steps.
    preprocessor_obj : ColumnTransformer
        The name of the preprocessor (column transformer) as per the Pipeline object definition

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with one column:
        - 'Coefficients': Lists the estimated feature coefficients for the input regression model.
        whose index is the column names of the input data frame of test samples (X)
        
    Examples:
    --------
    >>> import pandas as pd
    >>> from sklearn.pipeline import make_pipeline 
    >>> from sklearn.linear_model import Lasso, Ridge, LinearRegression
    >>> from sklearn.preprocessing import StandardScaler
    # Assume preprocessor is defined as necessary
    # Replace Lasso model with desired regression model (e.g. Ridge, LinearRegression)
    >>> pipe = make_pipeline(preprocessor, Lasso) 
    >>> pipe.fit(X_train, y_train)
    >>> results_df = show_feat_coeff(pipe, 'lasso', preprocessor)
    >>> results_df
    
    Notes:
    -----
    This function uses the pandas library to return result as DataFrame.

    """
    # Access .coef_ attribute of regression model 
    coef_vals = pipe_obj.named_steps[model_name].coef_
    coef_id = pd.Series(data=preprocessor_obj.get_feature_names_out()).apply(lambda x: x.split('__')[-1])
    # Report coefficient values and corresponding feature names in a DataFrame
    result = pd.DataFrame(
        data=coef_vals, index=coef_id, columns=["Coefficients"]
        ).sort_values(by="Coefficients", ascending=False)
    return result