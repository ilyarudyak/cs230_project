import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.special


class ReviewRNN(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax. Be careful to ensure your dimensions are correct after each step.

    You are encouraged to have a look at the network in pytorch/vision/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available to you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, dataset_params, training_params):
        """
        We define an recurrent network that predicts the NER tags for each token in the sentence. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            training_params: (Params) contains vocab_size, embedding_dim, lstm_hidden_dim
        """
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=dataset_params.vocab_size,
                                      embedding_dim=training_params.embedding_dim)
        self.lstm = nn.LSTM(input_size=training_params.embedding_dim,
                            hidden_size=training_params.lstm_hidden_dim,
                            num_layers=training_params.n_layers,
                            dropout=training_params.dropout,
                            batch_first=True)
        self.dropout = nn.Dropout(p=training_params.dropout)
        self.fc = nn.Linear(in_features=training_params.lstm_hidden_dim,
                            out_features=dataset_params.number_of_tags)
        
    def forward(self, x):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            x: (Variable) contains a batch of sentences, of dimension batch_size x seq_len, where seq_len is
               the length of the longest sentence in the batch. For sentences shorter than seq_len, the remaining
               tokens are PADding tokens. Each row is a sentence with each element corresponding to the index of
               the token in the vocab.

        Returns:
            out: (Variable) dimension batch_size*seq_len x num_tags with the log probabilities of tokens for each token
                 of each sentence.

        Note: the dimensions after each step are provided
        """
        batch_size = x.shape[0]

        #                                -> batch_size x seq_len
        # apply the embedding layer that maps each token to its embedding
        x_embed = self.embedding(x)            # dim: batch_size x seq_len x embedding_dim

        # run the LSTM along the sentences of length seq_len
        x_lstm, _ = self.lstm(x_embed)              # dim: batch_size x seq_len x lstm_hidden_dim

        # make the Variable contiguous in memory (a PyTorch artefact)
        hidden_dim = x_lstm.shape[2]
        x_flat = x_lstm.contiguous()

        # reshape the Variable so that each row contains one token
        x_flat = x_flat.view(-1, hidden_dim)       # dim: batch_size x seq_len x lstm_hidden_dim

        # apply the fully connected layer and obtain the output (before softmax) for each token
        x_out = self.fc(x_flat)                   # dim: batch_size x seq_len x num_tags
        output_size = x_out.shape[1]

        x_out = x_out.view(batch_size, -1, output_size)
        x_out = x_out[:, -1, :]

        return x_out   # dim: batch_size x seq_len x num_tags


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs from the model and labels for all tokens. Exclude loss terms
    for PADding tokens.

    Args:
        outputs: (Variable) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (Variable) dimension batch_size x seq_len where each element is either a label in [0, 1, ... num_tag-1],
                or -1 in case it is a PADding token.

    Returns:
        loss: (Variable) cross entropy loss for all tokens in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0).float()

    # indexing with negative values is not supported. Since PADded tokens have label -1, we convert them to a positive
    # number. This does not affect training, since we ignore the PADded tokens with the mask.
    labels = labels % outputs.shape[1]

    num_tokens = int(torch.sum(mask).item())

    # compute cross entropy loss for all tokens (except PADding tokens), by multiplying with mask.
    return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens
    
    
def accuracy(x_out, y):
    """
    Compute the accuracy, given the outputs and labels for all tokens.

    Args:
        x_out: (np.ndarray) dimension batch_size x num_tags - un-normalized output of the model
        y: (np.ndarray) dimension batch_size x 2 where each element is one hot encoded 0 or 1

    Returns: (float) accuracy in [0,1]
    """
    batch_size = x_out.shape[0]

    x_probs = scipy.special.softmax(x_out, axis=1)
    y_out = np.argmax(x_probs, axis=1)

    y_flat = np.argmax(y, axis=1)

    # compare outputs with labels and divide by batch_size
    return np.sum(y_out == y_flat) / float(batch_size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
