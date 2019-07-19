inv_vocab = defaultdict(list)
for key, value in vocab.items():
    inv_vocab[value].append(key)