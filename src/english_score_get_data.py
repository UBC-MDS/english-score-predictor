import pandas as pd
from sklearn.model_selection import train_test_split
import click
import os


@click.command()
@click.option("--url", required=True, type=str, help="URL of the dataset to download.")
@click.option(
    "--output_folder_path",
    required=True,
    type=str,
    help="Path to the output folder where files will be saved.",
)
def main(url, output_folder_path, sample_size=0.3, test_size=0.3):
    """
    Downloads, samples, splits, and saves a dataset.

    This function downloads a dataset from a specified URL using pandas.
    It samples a fraction of the dataset (default: 30%) and saves this sampled dataset
    to a ZIP file. It then splits the sampled dataset into training and testing subsets
    (default test size: 30%), and saves these subsets to separate CSV files in the specified
    directory. The directory is created if it does not exist.

    Parameters:
    url (str): URL of the dataset to download.
    output_folder_path (str): Path to the output folder where files will be saved.
    sample_size (float, optional): Fraction of the original dataset to sample.
                                   Defaults to 0.3.
    test_size (float, optional): Fraction of the sampled dataset to allocate
                                 for the test set. Defaults to 0.3.

    Returns:
    None: This function does not return anything but saves the sampled, training,
          and testing datasets as CSV files.
    """

    # Ensure the directory exists
    directory = output_folder_path
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Download and sample the dataset
    dataset = pd.read_csv(url, sep=",", on_bad_lines="skip", low_memory=False)
    sampled_dataset = dataset.sample(frac=sample_size, random_state=123)

    # Splitting the DataFrame
    train_df, test_df = train_test_split(
        sampled_dataset, test_size=test_size, random_state=123
    )

    # Save the training and testing datasets
    train_file_path = os.path.join(directory, "train_dataset.csv")
    test_file_path = os.path.join(directory, "test_dataset.csv")
    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)

    # Save the sampled dataset directly to a ZIP file
    zip_file_path = os.path.join(directory, "sampled_dataset.zip")
    compression_opts = dict(method="zip", archive_name="sampled_dataset.csv")
    sampled_dataset.to_csv(zip_file_path, index=False, compression=compression_opts)


if __name__ == "__main__":
    main()
