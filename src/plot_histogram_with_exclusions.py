import pandas as pd

def plot_histogram_with_exclusions(df, columns_to_exclude=None, bin=50, fig_size=(20, 15)):
    """
    Plots histograms for numeric columns in the provided DataFrame, with an option to exclude specific columns.

    This function filters the numeric columns in the DataFrame and generates a histogram for each one,
    except for those specified to be excluded. It is designed to handle larger datasets by enabling
    Altair's 'vegafusion' data transformer.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.
    columns_to_exclude : list of str, optional
        A list of column names to exclude from the histogram plots. If a column name specified does not exist,
        it will be ignored. Defaults to None, in which case no columns are excluded.
    bin : int, optional
        The number of bins to use for the histograms. Defaults to 50.
    fig_size : tuple of int, optional
        The size of the figure for each histogram, specified as (width, height). Defaults to (20, 15).

    Returns
    -------
    matplotlib.AxesSubplot or numpy.ndarray of them
        One or several matplotlib AxesSubplot objects depending on the number of numeric columns
        in the DataFrame, each representing a histogram for a particular column.

    Raises
    ------
    TypeError
        If the input `df` is not a pandas DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1], 'C': ['a', 'b', 'c', 'd', 'e']})
    >>> plot_histogram_with_exclusions(df, columns_to_exclude=['B'])
    >>> plt.show()  # Do not forget to import matplotlib.pyplot as plt to display the plots.

    Notes
    -----
    The function uses matplotlib as the underlying plotting library, which integrates with pandas.
    The histograms will be displayed immediately if running in a Jupyter notebook. In a script,
    you may need to call `plt.show()` to display the figures.
    """
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Select numeric columns and exclude specified columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if columns_to_exclude:
        numeric_cols = [col for col in numeric_cols if col not in columns_to_exclude]


    return df[numeric_cols].hist(bins=bin, figsize=fig_size)