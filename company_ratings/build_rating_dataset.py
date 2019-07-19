"""Read, split and save the kaggle dataset for our model"""

import os
from argparse import Namespace
import pandas as pd

args = Namespace(
    train_share=.8,
    val_share=.1,
    test_share=.1,
    employee_reviews_prep_5w='data/employee_reviews_prep_5w.csv',
    data_dir='data/5w'
)


def build_dataset():

    df = pd.read_csv(args.employee_reviews_prep_5w)

    data_size = df.shape[0]
    train_upper_bound = int(args.train_share * data_size)
    val_upper_bound = int((args.train_share + args.val_share) * data_size)

    # write train files
    train_path_review = os.path.join(args.data_dir, 'train/review.txt')
    train_path_rating = os.path.join(args.data_dir, 'train/rating.txt')
    df.iloc[0:train_upper_bound].review.to_csv(train_path_review, index=False)
    df.iloc[0:train_upper_bound].rating.to_csv(train_path_rating, index=False)

    # write val files
    val_path_review = os.path.join(args.data_dir, 'val/review.txt')
    val_path_rating = os.path.join(args.data_dir, 'val/rating.txt')
    df.iloc[train_upper_bound:val_upper_bound].review.to_csv(val_path_review, index=False)
    df.iloc[train_upper_bound:val_upper_bound].rating.to_csv(val_path_rating, index=False)

    # write test files
    test_path_review = os.path.join(args.data_dir, 'test/review.txt')
    test_path_rating = os.path.join(args.data_dir, 'test/rating.txt')
    df.iloc[val_upper_bound:].review.to_csv(test_path_review, index=False)
    df.iloc[val_upper_bound:].rating.to_csv(test_path_rating, index=False)


if __name__ == "__main__":
    build_dataset()
