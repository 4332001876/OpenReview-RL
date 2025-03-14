"""
Preprocess the OpenReview dataset to parquet format
"""

import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import datasets

def load_openreview_data(data_path, abstract_col='abstract', rating_col='ratings_avg'):
    """
    Load OpenReview data from a CSV file
    
    Args:
        data_path (str): Path to the CSV file
        abstract_col (str): Name of the abstract column
        rating_col (str): Name of the rating column
        
    Returns:
        pd.DataFrame: Dataframe with abstract and rating columns
    """
    df = pd.read_csv(data_path)
    print(df.columns)

    # Ensure required columns exist
    if abstract_col not in df.columns:
        raise ValueError(f"Abstract column '{abstract_col}' not found in data")
    if rating_col not in df.columns:
        raise ValueError(f"Rating column '{rating_col}' not found in data")
    
    # Keep only relevant columns and rename them
    df = df[[abstract_col, rating_col]].rename(
        columns={abstract_col: 'abstract', rating_col: 'rating'}
    )
    
    # Drop rows with missing values
    df = df.dropna()
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/openreview')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', required=True, help='Path to the OpenReview CSV file')
    parser.add_argument('--abstract_col', default='abstract', help='Column name for abstracts')
    parser.add_argument('--rating_col', default='ratings_avg', help='Column name for ratings')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for train/test split')

    args = parser.parse_args()

    # Load the dataset
    df = load_openreview_data(args.data_path, args.abstract_col, args.rating_col)
    
    # Split the dataset into train and test
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state
    )
    
    # Convert to datasets format
    train_dataset = datasets.Dataset.from_pandas(train_df)
    test_dataset = datasets.Dataset.from_pandas(test_df)

    # Process each data item
    def make_map_fn(split):
        def process_fn(example, idx):
            data_source = 'openreview'
            instruction_following = (
                r'Review the paper abstract and rate its quality from 1 to 10, '
                r'where 1 is very poor and 10 is excellent. '
                r'Your response should be divided into two parts: first, you think about the evaluation process as an internal monologue, enclosed within <think> </think> tags, '
                r'and second, your final rating within <answer> </answer> tags, containing only an integer from 1 to 10.'
            )
            
            abstract = example.pop('abstract')
            rating = example.pop('rating')
            prompt = f"Paper Abstract: {abstract}\n\n{instruction_following}"
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "ability": "paper_review",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(rating)
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'rating': rating,
                    "abstract": abstract,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, num_proc=8)

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Save as parquet files
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
