import pandas as pd
from sklearn.model_selection import train_test_split
import click
import os

@click.command()
@click.option('--sample_size', type=float)
@click.option('--test_size', type=float)



def main(sample_size=0.3, test_size=0.3):
    """
    Main function for dataset sampling, splitting, and saving.

    This function downloads a dataset from a specified URL using pandas.
    It samples a fraction of the dataset based on the provided sample size and saves 
    this sampled dataset to a CSV file. It then splits the sampled dataset into 
    training and testing subsets based on the provided test size, and saves these 
    subsets to separate CSV files in the '../data/raw' directory. The directory 
    is created if it does not exist.

    Parameters:
    sample_size (float, optional): Fraction of the original dataset to sample. 
                                   Defaults to 0.3.
    test_size (float, optional): Fraction of the sampled dataset to allocate 
                                 for the test set. Defaults to 0.3.

    Returns:
    None: This function does not return anything but saves the sampled, training, 
          and testing datasets as CSV files.
    """

    # Ensure the directory exists
    directory = "./data/raw"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Download and sample the dataset
    dataset = pd.read_csv(
        "https://osf.io/download/g72pq/", sep=",", on_bad_lines="skip", low_memory=False
    )
    sampled_dataset = dataset.sample(frac=sample_size, random_state=123)
    sampled_file_path = os.path.join(directory, "sampled_dataset.csv")
    sampled_dataset.to_csv(sampled_file_path, index=False)
    
    # Splitting the DataFrame
    train_df, test_df = train_test_split(sampled_dataset, test_size=test_size, random_state=123)

    # Save the training and testing datasets
    train_file_path = os.path.join(directory, "train_dataset.csv")
    test_file_path = os.path.join(directory, "test_dataset.csv")
    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)




if __name__ == '__main__':
    main()
