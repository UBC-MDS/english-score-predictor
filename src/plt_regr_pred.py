import matplotlib.pyplot as plt
import numpy as np


def plt_regr_pred(X, y, pipe_obj):
    """
    Plot the actual target values against the predicted target values based on training
    data used to fit a pipeline of transformers with a regression model estimator.

    Parameters:
    ----------
    pipe_obj : Pipeline
        The fitted input pipeline of transformers with a regression model estimator. Pipeline must be fitted to 
        be able to call `predict`. 
    X : array-like shape (n_samples, n_features)
        The input array-like shape containing the training data on which the input Pipeline object was fit.
        This training data will be used to predict the target values using the Pipeline object.
    y : array-like of shape (n_samples,)
        The input array-like shape of n_samples number of entries containing the target values associated with the
        training data X and on which the Pipeline object was fit. These are the actual target values.

    Returns:
    -------
    None
        Plot of actual target values against predictied target values will be shown as output of calling function.
        
    Examples:
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> # Assume preprocessor object has been defined with appropriate feature transformations 
    # Replace Lasso model with desired regression model (e.g. Ridge, LinearRegression)
    >>> pipe = Pipeline([preprocessor, ('lasso', Lasso())]) 
    >>> pipe.fit(X_train, y_train)
    >>> plt_regr_pred(X_train, y_train, pipe)
    
    Notes:
    -----
    This function uses the pandas library linspace function to help make the diagonal.
    This function also uses the matplotlib pyplot module for plotting.

    """
    plt.scatter(y, pipe_obj.predict(X), alpha=0.3)
    grid = np.linspace(y.min(), y.max(), 1000)
    plt.plot(grid, grid, "--k")
    plt.xlabel("Actual Target")
    plt.ylabel("Predicted Target")
    plt.show()
    return None