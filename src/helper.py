# Python file to define helper functions
import pandas as pd
import altair as alt

# Example function
def sum(a, b):
    """Function to sum two numbers"""
    return a + b


def plot_histogram_with_exclusions(df, columns_to_exclude=None, bins=50, figsize=(20, 15)):
    """
    Plot histograms for numeric columns in a DataFrame, with the option to exclude specified columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.
    columns_to_exclude : list of str, optional
        List of column names to exclude from the plot. If a column in this list does not exist in the DataFrame,
        it will be ignored. Defaults to None.
    bins : int, optional
        Number of bins for the histogram. Defaults to 50.
    figsize : tuple, optional
        Figure size for the histograms. Defaults to (20, 15).

    Returns
    -------
    altair.vegalite.v4.api.Chart
        Altair Chart object consisting of histograms for each numeric column not excluded.

    Examples
    --------
    >>> train_df = pd.DataFrame(...)
    >>> plot_histogram_with_exclusions(train_df, columns_to_exclude=['unwanted_column1', 'unwanted_column2'])
    
    Notes
    -----
    This function uses Altair for plotting, which should be enabled with 'alt.data_transformers.enable("vegafusion")'
    to handle larger datasets more efficiently.
    """

    # Filter out non-existing columns from the columns_to_exclude list
    if columns_to_exclude is not None:
        columns_to_exclude = [col for col in columns_to_exclude if col in df.columns]

    # Drop specified columns
    df = df.drop(columns=columns_to_exclude, errors='ignore')

    # Enable Altair data transformer
    alt.data_transformers.enable("vegafusion")

    # Select numeric columns and plot histograms
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    charts = [alt.Chart(df).mark_bar().encode(
                alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins)),
                y='count()',
                ).properties(
                width=figsize[0],
                height=figsize[1],
                title=f'Histogram of {col}'
                ) for col in numeric_cols]

    return alt.vconcat(*charts)

