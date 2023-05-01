import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
import glob
import random
import os
from typing import Tuple
import time
import math
import wandb

import build_vocab
import transformer_models as tf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_files(path): return glob.glob(path)

class Transformer_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sequence_length,
        batch_size,
        vocab_file,
        token_file
    ):
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self.load_words(vocab_file, token_file)
    
        self.uniq_words = len(self.vocab)

    def load_words(self, vocab_file, token_file):
        self.vocab = build_vocab.load_vocab(find_files(vocab_file)[0])

        token_file = find_files(token_file)

        self.raw_tokens = build_vocab.load_tokenized_file(token_file[0])

        self.num_samples = len(self.raw_tokens)

        self.num_sequences = max(1, (self.num_samples // self.sequence_length))
        self.num_batches = max(1, self.num_sequences // self.batch_size)
    
        print('Number of raw_tokens: ', len(self.raw_tokens))
        print('Number of samples in a batch: ', self.num_sequences)
        print('Number of batches: ', self.num_batches)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        index = index * self.sequence_length
        # TODO: make sure this is returning correctly
        # We cut off the potential of it being too long
        #random_index = random.randint(0, len(current_sample) - (self.sequence_length + 1)) 
        #end_index = random_index + self.sequence_length
        return ( # might break if it gets the very end?
            torch.tensor(self.raw_tokens[index:(index + self.sequence_length)]).to(device), # x
            torch.tensor(self.raw_tokens[index+1:(index + self.sequence_length+1)]).to(device) # y
        )
    
# Define all hyperparameters and our model

def train(model: nn.Module,
          dataset: torch.utils.data.Dataset,
          batch_size: int,
          sequence_length: int,
          num_epochs: int,
          ntokens: int,
          lr: float) -> None:
    model.train()  # turn on train mode
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) # Could try adam
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    total_loss = 0.
    log_interval = dataset.num_batches - 1
    start_time = time.time()
    src_mask = tf.generate_square_subsequent_mask(sequence_length).to(device)

    num_batches = dataset.num_batches

    for epoch in range(num_epochs):

        for batch, (x, y) in enumerate(dataloader):
            #print('X size: ', x.size()) # 16, 256 - batch_size, sequence_length
            output = model(x, src_mask)

            # Not sure why we reshape output, will have to mess with this
            loss = criterion(output.transpose(1, 2), y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                ppl = math.exp(cur_loss)
                print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                total_loss = 0
                start_time = time.time()

# TODO: Implement evaluation method
# In order to implement, I need to create an eval data set and add a way to get a batch from the dataloader for that
# Just using this for evaluation, to take input data and create target tensor
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    pass
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    """seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target"""


"""def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = tf.generate_square_subsequent_mask(sequence_length).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, sequence_length):
            data, targets = get_batch(eval_data, i)
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += sequence_length * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)"""

def predict(model: nn.Module, input: str, dataset, generation_length=500):

    # Format the input
    prediction = build_vocab.get_vocab_indx_vector(dataset.vocab, build_vocab.load_spacy(), input)

    # Tokenize input
    sequence = torch.tensor(prediction).to(device)

    # Format tensor to be in shape: batch_size, seq_len
    sequence = sequence.reshape(1, -1)

    for _ in range(generation_length):
        # Give transformer entire sequence
        output = model(sequence)

        #print(output[0,-1].size())
        # Take last generated sequence of highest probability
        log_prob_vals, next_word = torch.topk(output[0,-1], k=1, dim=0)

        #print(log_prob_vals)
        #print(next_word)
        # append to sequence
        prediction.append(next_word.item())
        sequence = torch.tensor(prediction).to(device).reshape(1, -1)

        # Take context window and feed back in?
    
    #print(prediction)

    # Decode entire sequence
    final_prediction = build_vocab.decode_vocab(dataset.vocab, prediction)

    return final_prediction



# This is a way to perform multiple runs in the same script.
def train_wrapper():
    vocab_file = 'C:\\Users\\jayor\\Documents\\repos\\controllableRNN\\vocabs_and_tokens\\books\\gatsby_vocab.pt'
    token_file = 'C:\\Users\\jayor\\Documents\\repos\\controllableRNN\\vocabs_and_tokens\\books\\gatsby_tok.pkl'
    
    # Create dataset
    sequence_length = 256 # Length of one sequence
    batch_size = 16 # Nymber of sequences in a batch
    tag_type = 'languages' # Dummy for now because we aren't using it
    dataset = Transformer_Dataset(sequence_length, batch_size, vocab_file, token_file)

    ntokens = dataset.uniq_words  # size of vocabulary
    emsize = 300  # embedding dimension
    d_hid = 900  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 9  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 3  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    model = tf.TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    lr = 5.0  # learning rate
    num_epochs = 10000

    train(model, dataset, batch_size, sequence_length, num_epochs, ntokens, lr)

    input = '“How could she of been like that?”'
    prediction = predict(model, input, dataset)

    print(' '.join(prediction))


def main():
    #wandb.login()
    train_wrapper()


if __name__ == "__main__":
    main()