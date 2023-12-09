import matplotlib.pyplot as plt
import numpy as np


def plt_regr_pred(X, y, pipe_obj):
    """
    Plot a 2D histogram of the actual target values against the predicted target values based on training
    data used to fit a pipeline of transformers with a regression model estimator.

    Parameters:
    ----------
    pipe_obj : Pipeline
        The fitted input pipeline of transformers with a regression model estimator. Pipeline must be fitted to 
        be able to call `predict`. 
    X : pandas.DataFrame (n_samples, n_features)
        The input data frame containing the training data on which the input Pipeline object was fit.
        This training data will be used to predict the target values using the Pipeline object.
    y : pandas.DataFrame (n_samples,)
        The input data frame shape of n_samples number of entries containing the target values associated with the
        training data X and on which the Pipeline object was fit. These are the actual target values.

    Returns:
    -------
    matplotlib Axes object
        Returns the Axes object representing the subplot of actual target values against predictied target values. 
        Plot will be shown as output of calling function.
        
    Examples:
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from sklearn.linear_model import Lasso, Ridge, LinearRegression
    >>> from sklearn.preprocessing import StandardScaler
    # Replace StandardScaler with desired preprocessing step(s)
    # Replace Lasso model with desired regression model (e.g. Ridge, LinearRegression)
    >>> pipe = Pipeline([('stdsclr', StandardScaler()), ('lasso', Lasso())]) 
    >>> pipe.fit(X_train, y_train)
    >>> plt_regr_pred(X_train, y_train, pipe)
    
    Notes:
    -----
    This function uses the pandas library linspace function to help make the diagonal.
    This function also uses the matplotlib pyplot module for plotting.

    """
    # Creating new figure and set of subplots
    fig, ax = plt.subplots()
    # Plotting a 2D histogram
    H = ax.hist2d(y, pipe_obj.predict(X), bins=50, cmap='viridis')
    # Plot the diagonal ("perfect prediction" line)
    grid = np.linspace(y.min(), y.max(), 1000)
    ax.plot(grid, grid, "--k")
    # Add labels and show plot
    ax.set_xlabel("Actual Target")
    ax.set_ylabel("Predicted Target")
    ax.set_title("2D Histogram of Actual vs. Predicted Target Values")
    plt.colorbar(H[3], ax=ax, label='Frequency')
    plt.show()

    return ax