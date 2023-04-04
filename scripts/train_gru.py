import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import os
import glob
import sys

import gru_models
import build_vocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE=16
TRAIN_TOKEN_LEN=256

#from vocab_building import load_tokenized_file, load_vocab, decode_vocab, nlp, get_vocab_indx_vector

def find_files(path): return glob.glob(path)

class RNN_Dataset_multiple_sources(torch.utils.data.Dataset):
    def __init__(
        self,
        sequence_length,
        type
    ):
        folder = "../vocabs_and_tokens/" + type + "/"
        data_folder = "../data/" + type + "/"
        vocab_file = folder + "*.pt"
        token_files = folder + "*.pkl"
        self.sequence_length = sequence_length

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
        self.vocab = build_vocab.load_vocab(find_files(vocab_file)[0])

        token_files = find_files(token_files)
        # This is only setup to handle two different categories right now
        self.raw_tokens_1 = build_vocab.load_tokenized_file(token_files[0])
        self.raw_tokens_2 = build_vocab.load_tokenized_file(token_files[1])

        self.num_samples_1 = len(self.raw_tokens_1)
        self.num_samples_2 = len(self.raw_tokens_2)

        # This is iffy, because we aren't actually going through all of the "samples"
        self.num_samples = max(1, ((self.num_samples_1 + self.num_samples_2) // TRAIN_TOKEN_LEN)) # Split raw tokens into groups of TRAIN_TOKEN_LEN
        self.num_batches = max(1, self.num_samples // BATCH_SIZE)

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

def train(dataset, model, max_epochs, batch_size, cat = False):
    train_losses = []

    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(max_epochs):

        total_loss = 0
        
        for batch, (x, y, category) in enumerate(dataloader):
            hidden_states = model.init_hidden(batch_size)

            #print('x size: ', x.size()) # 16, 256
            #print('category size: ', category.size()) # 16, 256
            
            optimizer.zero_grad()

            if cat:
                y_pred, hidden_states = model(x, hidden_states, batch_size, category)
            else:
                y_pred, hidden_states = model(x, hidden_states, batch_size)

            #print('y_pred size: ', y_pred.size()) # [16, 4822] for cells, [16, 256, 4822] normal
            #print('y_pred transposed size: ', y_pred.transpose(1, 2).size()) # [16, 4822, 256]

            loss = criterion(y_pred.transpose(1, 2), y)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })

        train_losses.append(total_loss/batch_size)

    return train_losses

def predict_with_category(dataset, model, text, category, next_words=100):
    model.eval()

    prediction = build_vocab.get_vocab_indx_vector(dataset.vocab, build_vocab.load_spacy, text)
    tokens = torch.tensor(prediction).to(device)

    # Get category tensor
    li = dataset.all_categories.index(category)
    if li == 0:
        category = torch.zeros(len(prediction)).to(device).long()
    else:
        category = torch.ones(len(prediction)).to(device).long()

    print('cat size: ', category.size())
    print('prediction size: ', tokens.size())

    state_h = model.init_hidden(1) # num_layers, batch_size, lstm_size

    # Prime generation by feeding in initial input:
    for p in range(len(tokens)-1):
        _, state_h = model(tokens[p].view(1,-1), state_h, 1, category[p].view(1,-1))
        #print('state_h size: ', state_h.size())

    last_token = tokens[-1]
    for i in range(0, next_words):
        y_pred, state_h = model(last_token.view(1,-1), state_h, 1, category[0].view(1,-1))
        #print('y_pred size: ', y_pred.size()) # [16, 256, 12923], should be [1, 1, 12923]
        #print('y_pred[0][-1] size: ', y_pred[0][-1].size())

        last_word_logits = y_pred[0][-1]

        # These are the probabilities
        p = torch.nn.functional.softmax(last_word_logits, dim=0)
        word_index = torch.multinomial(p, 1)[0]
        top_values = torch.topk(p, 5)
        #top_words = top_values.indices
        #top_probs = top_values.values

        #print('word index: ', word_index)
        #print('top_words: ', top_words.tolist())
        #top_word_pred = decode_vocab(dataset.vocab, [word_index])
        #top_words_pred = decode_vocab(dataset.vocab, top_words.tolist())

        #print('The top word predicted was: ', top_word_pred)
        #print('The top five predictions were: ', top_words_pred)
        #print('Their probabilites are: ', top_probs)

        prediction.append(word_index)

        last_token = torch.tensor([word_index]).to(device)

    final_prediction = build_vocab.decode_vocab(dataset.vocab, prediction)
    return final_prediction


def main():
    arguments = sys.argv[1:]
    type, num_epochs, hidden_size = arguments
    num_epochs = int(num_epochs)
    hidden_size = int(hidden_size)

    # Create dataset
    dataset = RNN_Dataset_multiple_sources(TRAIN_TOKEN_LEN, type)
    input_size = dataset.uniq_words # Should be size of vocab?
    n_layers = 3

    # Model with normal pytorch GRU
    category_model = gru_models.GRU_category(input_size, hidden_size, input_size, n_layers).to(device)

    file_path = f"gru_trained_cat_reviews.pt"

    losses_cat = train(dataset, category_model, num_epochs, BATCH_SIZE, cat=True)

    torch.save(category_model.state_dict(), file_path)
    
    # Model with GRU Cells
    cells_category_model = gru_models.GRU_with_cells_category(input_size, hidden_size, input_size, n_layers).to(device)

    file_path = f"gru_trained_cat_cells_reviews.pt"

    losses_cat_cells = train(dataset, cells_category_model, num_epochs, BATCH_SIZE, True)

    torch.save(cells_category_model.state_dict(), file_path)

    # Model with edited GRU Cells
    cells_category_edited_model = gru_models.GRU_with_cells_category_edited(input_size, hidden_size, input_size, n_layers).to(device)

    file_path = f"gru_trained_cat_cells_edited_reviews.pt"

    losses_cat_cells_edited = train(dataset, cells_category_edited_model, num_epochs, BATCH_SIZE, True)

    torch.save(cells_category_edited_model.state_dict(), file_path)

    # Create loss graph and save
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(losses_cat)), losses_cat, label="original")
    ax.plot(range(len(losses_cat_cells)), losses_cat_cells, label="original with cells")
    ax.plot(range(len(losses_cat_cells_edited)), losses_cat_cells_edited, label="edited")
    plt.title("Loss over time")
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('loss_' + str(type) + "_" + str(num_epochs) + "_" + str(hidden_size) + '.png')

if __name__ == "__main__":
    main()
