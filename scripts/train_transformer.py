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
        tag_type
    ):
        folder = "../vocabs_and_tokens/" + tag_type + "/"
        data_folder = "../data/" + tag_type + "/"
        vocab_file = folder + "*.pt"
        token_files = folder + "*.pkl"
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self.all_categories, self.n_categories = self.setup_categories(data_folder)
        self.load_words(vocab_file, token_files)
    
        self.uniq_words = len(self.vocab)

    # data_folder needs to be like '../data/reviews/'
    def setup_categories(self, data_folder):
        all_categories = []
        for filename in find_files(data_folder + '*.txt'):
            category = os.path.splitext(os.path.basename(filename))[0]
            all_categories.append(category)
            
        n_categories = len(all_categories)

        if n_categories == 0:
            raise RuntimeError('Data not found.')

        print('# categories:', n_categories, all_categories)
        #all_categories.remove('garden')
        #all_categories.remove('music')
        #all_categories.remove('small_combined')
        #n_categories_languages = len(all_categories)
        #print('# categories:', n_categories_languages, all_categories)

        return all_categories, n_categories

    def load_words(self, vocab_file, token_files):
        # We want the vocab to be constructed from all sources, but we need the raw token sets for each seperately.
        # The category vector can just be a simple index vector.
        print(vocab_file)
        self.vocab = build_vocab.load_vocab(find_files(vocab_file)[0])

        token_files = find_files(token_files)
        # This is only setup to handle two different categories right now
        self.raw_tokens_1 = build_vocab.load_tokenized_file(token_files[0])
        self.raw_tokens_2 = build_vocab.load_tokenized_file(token_files[1])

        self.num_samples_1 = len(self.raw_tokens_1)
        self.num_samples_2 = len(self.raw_tokens_2)

        # This is iffy, because we aren't actually going through all of the "samples"
        self.num_samples = max(1, ((self.num_samples_1 + self.num_samples_2) // self.sequence_length)) # Split raw tokens into groups of TRAIN_TOKEN_LEN
        self.num_batches = max(1, self.num_samples // self.batch_size)

        print('Number of raw_tokens: ', len(self.raw_tokens_1 + self.raw_tokens_2))
        print('Number of samples in a batch: ', self.num_samples)
        print('Number of batches: ', self.num_batches)

        return 1
    
    def random_choice(self, l):
        return l[random.randint(0, len(l)-1)]
    
    def category_tensor(self, category):
        li = self.all_categories.index(category)
        if li == 0:
            tensor = torch.zeros(self.sequence_length).to(device).long()
        else:
            tensor = torch.ones(self.sequence_length).to(device).long()
        return tensor, li

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # This should pick a random source, grab it's category, and then grab a sequence associated with it.
        # Pick random category
        string_category= self.random_choice(self.all_categories)
        category, category_index = self.category_tensor(string_category)

        # Pick the right token samples based on the category
        if category_index == 0:
            current_sample = self.raw_tokens_1
        else:
            current_sample = self.raw_tokens_2
            
        # We cut off the potential of it being too long
        random_index = random.randint(0, len(current_sample) - (self.sequence_length + 1)) 
        end_index = random_index + self.sequence_length
        return ( # might break if it gets the very end?
            torch.tensor(current_sample[random_index:end_index]).to(device), # x
            torch.tensor(current_sample[random_index+1:end_index+1]).to(device), # y
            category
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
    log_interval = 200
    start_time = time.time()
    src_mask = tf.generate_square_subsequent_mask(sequence_length).to(device)

    num_batches = dataset.num_batches

    for epoch in range(num_epochs):

        #for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        for batch, (x, y, category) in enumerate(dataloader):
            output = model(x, src_mask)

            # Not sure why we reshape output, will have to mess with this
            loss = criterion(output.view(-1, ntokens), y)

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
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = tf.generate_square_subsequent_mask(sequence_length).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, sequence_length):
            data, targets = get_batch(eval_data, i)
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += sequence_length * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)

# This is a way to perform multiple runs in the same script.
def train_wrapper():
    
    # Create dataset
    sequence_length = 256 # Length of one sequence
    batch_size = 16 # Nymber of sequences in a batch
    tag_type = 'languages' # Dummy for now because we aren't using it
    dataset = Transformer_Dataset(sequence_length, batch_size, tag_type)

    ntokens = dataset.uniq_words  # size of vocabulary
    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 3  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    model = tf.TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    lr = 5.0  # learning rate
    num_epochs = 50

    train(model, dataset, batch_size, sequence_length, num_epochs, ntokens, lr)


def main():
    wandb.login()
    train_wrapper()


if __name__ == "__main__":
    main()