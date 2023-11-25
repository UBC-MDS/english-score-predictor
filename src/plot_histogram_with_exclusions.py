import pandas as pd
import altair as alt

def plot_histogram_with_exclusions(df, columns_to_exclude=None, bins=50, figsize=(20, 15)):
    """
    Plot histograms for each numeric column in a DataFrame, excluding specified columns.

    This function creates a histogram for each numeric column in the provided DataFrame,
    allowing for specified columns to be excluded from the plot. It uses Altair for
    visualization, which is well-suited for handling large datasets.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to be plotted.
    columns_to_exclude : list of str, optional
        A list of column names to be excluded from the histogram plots.
        If a column in this list does not exist in the DataFrame, it will be ignored.
        Defaults to None, meaning no columns will be excluded.
    bins : int, optional
        The number of bins to use for the histograms. Defaults to 50.
    figsize : tuple of int, optional
        The size of the figure for the histograms, provided as a tuple (width, height).
        Defaults to (20, 15).

    Returns
    -------
    altair.vegalite.v4.api.VConcatChart
        An Altair VConcatChart object that displays the histograms for each numeric column
        not excluded.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1], 'C': ['a', 'b', 'c', 'd', 'e']})
    >>> histogram_plot = plot_histogram_with_exclusions(df, columns_to_exclude=['B'])
    >>> histogram_plot.display()

    Notes
    -----
    To handle larger datasets efficiently, this function enables Altair's 'vegafusion' data transformer.
    """

    # Enable Altair data transformer for efficient handling of larger datasets
    alt.data_transformers.enable("vegafusion")
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Select numeric columns and exclude specified columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if columns_to_exclude:
        numeric_cols = [col for col in numeric_cols if col not in columns_to_exclude]

    # Create histograms for each numeric column
    charts = [
        alt.Chart(df).mark_bar().encode(
            alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins)),
            y='count()'
        ).properties(
            width=figsize[0],
            height=figsize[1],
            title=f'Histogram of {col}'
        ) for col in numeric_cols
    ]

    # Concatenate all charts vertically
    return alt.vconcat(*charts)
