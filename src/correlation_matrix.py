import pandas as pd

def pearson_correlation_matrix(df, colormap='seismic'):
    """Generate a styled Pearson correlation matrix plot for selected features.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing numeric columns for correlation analysis.
    colormap : str, optional
        The colormap for the background gradient styling of the correlation matrix.
        Defaults to 'seismic'.

    Returns
    -------
    pandas.io.formats.style.Styler
        A styled correlation matrix plot.

    Notes
    -----
    - This function selects numeric columns without null values from the input DataFrame.
    - Constant columns (columns with no variability) are excluded to avoid undefined correlations.
    - The resulting correlation matrix is styled with a background gradient using the specified colormap input.

    Examples
    --------
    >>> import pandas as pd
    >>> from your_module import pearson_correlation_matrix

    >>> # Load your DataFrame (e.g., train_df)
    >>> train_df = pd.read_csv('your_data.csv')

    >>> # Generate and display a Pearson correlation matrix plot
    >>> pearson_matrix = pearson_correlation_matrix(train_df, colormap='viridis')
    >>> display(pearson_matrix)
    """
    # Select numeric columns without null values
    df = df.dropna(axis=1)
    selected_features = df.select_dtypes(include='number').loc[
        :, df.select_dtypes(include='number').nunique() > 1
    ]

     # Identify excluded columns
    all_columns = set(df.select_dtypes(include='number').columns)
    excluded_columns = list(all_columns - set(selected_features.columns))

    # Print excluded columns
    print("Excluded columns:\n", excluded_columns)

    # Calculate the Pearson correlation matrix
    correlation_matrix = selected_features.corr()

    # Apply background gradient styling
    correlation_matrix_style = correlation_matrix.style.background_gradient(
        cmap=colormap
    )

    # Return the styled correlation matrix
    return correlation_matrix_style
