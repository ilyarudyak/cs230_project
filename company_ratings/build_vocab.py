"""Build vocabularies of words and tags from datasets"""

import argparse
from collections import Counter
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=3, help="Minimum count for words in the dataset", type=int)
parser.add_argument('--min_count_tag', default=1, help="Minimum count for tags in the dataset", type=int)
parser.add_argument('--data_dir', default='data/5w', help="Directory containing the dataset")

# Hyper parameters for the vocab
PAD_WORD = '<pad>'
PAD_TAG = 'O'
UNK_WORD = 'UNK'


def save_vocab_to_txt_file(vocab, txt_path):
    """Writes one token per line, 0-based line id corresponds to the id of the token.

    Args:
        vocab: (iterable object) yields token
        txt_path: (string) path to vocab file
    """
    with open(txt_path, "w") as f:
        for token in vocab:
            f.write(token + '\n')


def save_dict_to_json(d, json_path):
    """Saves dict to json file

    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def update_vocab(txt_path, vocab):
    """Update word and tag vocabulary from dataset

    Args:
        txt_path: (string) path to file, one sentence per line
        vocab: (dict or Counter) with update method

    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    with open(txt_path) as f:
        for i, line in enumerate(f):
            vocab.update(line.strip().replace('"', '').split(' '))

    return i + 1


def build_vocab(data_dir):
    # Build word vocab with train and test datasets
    print("Building word vocabulary...")
    vocab = Counter()
    size_train_sentences = update_vocab(os.path.join(data_dir, 'train/review.txt'), vocab)
    size_dev_sentences = update_vocab(os.path.join(data_dir, 'val/review.txt'), vocab)
    size_test_sentences = update_vocab(os.path.join(data_dir, 'test/review.txt'), vocab)
    print("- done.")

    # Build tag vocab with train and test datasets
    print("Building tag vocabulary...")
    tags = Counter()
    size_train_tags = update_vocab(os.path.join(data_dir, 'train/rating.txt'), tags)
    size_dev_tags = update_vocab(os.path.join(data_dir, 'val/rating.txt'), tags)
    size_test_tags = update_vocab(os.path.join(data_dir, 'test/rating.txt'), tags)
    print("- done.")

    # Assert same number of examples in datasets
    assert size_train_sentences == size_train_tags
    assert size_dev_sentences == size_dev_tags
    assert size_test_sentences == size_test_tags

    sizes = {
        'train_size': size_train_sentences,
        'dev_size': size_dev_sentences,
        'test_size': size_test_sentences,
        'number_of_tags': len(tags),
        'pad_word': PAD_WORD,
        'pad_tag': PAD_TAG,
        'unk_word': UNK_WORD
    }

    return vocab, tags, sizes


if __name__ == '__main__':
    args = parser.parse_args()

    vocab, tags, sizes = build_vocab(args.data_dir)
    print(len(vocab))

    # Only keep most frequent tokens
    words = [tok for tok, count in vocab.items() if count >= args.min_count_word]
    tags = [tok for tok, count in tags.items() if count >= args.min_count_tag]

    # Add pad tokens
    if PAD_WORD not in words: words.append(PAD_WORD)
    if PAD_TAG not in tags: tags.append(PAD_TAG)

    # add word for unknown words
    words.append(UNK_WORD)

    # Save vocabularies to file
    print("Saving vocabularies to file...")
    save_vocab_to_txt_file(words, os.path.join(args.data_dir, 'words.txt'))
    save_vocab_to_txt_file(tags, os.path.join(args.data_dir, 'tags.txt'))
    print("- done.")

    # Save datasets properties in json file
    sizes['vocab_size'] = len(words)
    save_dict_to_json(sizes, os.path.join(args.data_dir, 'dataset_params.json'))

    # Logging sizes
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))
