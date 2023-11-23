import pandas as pd


def pearson_correlation_matrix(df, colormap):
    """
    Generate and display a styled correlation matrix for the given DataFrame.

    Parameters:
    - df: DataFrame
    """
    # Select numeric columns without null values
    selected_features = df.select_dtypes(include="number").loc[
        :, df.select_dtypes(include="number").nunique() > 1
    ]
    print("These are the selected features for the correlation matrix plot\n",selected_features.columns)

    # Calculate the pearson correlation matrix
    correlation_matrix = selected_features.corr()

    # Apply background gradient styling
    correlation_matrix_style = correlation_matrix.style.background_gradient(
        cmap=colormap
    )

    # Display the styled correlation matrix
    return correlation_matrix_style
