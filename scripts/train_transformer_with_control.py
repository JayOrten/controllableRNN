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
import transformer_model_category
import transformer_model_category_edited_1
import transformer_model_category_edited_2
import transformer_model_category_edited_3
import transformer_model_category_edited_3_2
import transformer_model_category_edited_4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 

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
        self.spacy = build_vocab.load_spacy()

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

        return all_categories, n_categories

    def load_words(self, vocab_file, token_files):
        # We want the vocab to be constructed from all sources, but we need the raw token sets for each seperately.
        # The category vector can just be a simple index vector.
        self.vocab = build_vocab.load_vocab(find_files(vocab_file)[0])

        token_files = find_files(token_files)
        
        self.train_tokens = []
        self.train_num_sequences = []

        self.eval_tokens = []
        self.eval_num_sequences = []

        for token_file in token_files:
            raw_tokens = build_vocab.load_tokenized_file(token_file)
            test_split = int(len(raw_tokens) * .90)
 
            train_token = raw_tokens[:test_split]
            eval_token = raw_tokens[test_split:]

            num_seq = len(train_token)//self.sequence_length
            num_eval_seq = len(eval_token)//self.sequence_length
            
            self.train_tokens.append(train_token)
            self.train_num_sequences.append(num_seq)

            self.eval_tokens.append(eval_token)
            self.eval_num_sequences.append(num_eval_seq)
        
        # This is iffy, because we aren't actually going through all of the "samples"
        self.num_samples = max(1, (sum([len(tok) for tok in self.train_tokens]) // self.sequence_length)) # Split raw tokens into groups of TRAIN_TOKEN_LEN
        self.num_batches = max(1, self.num_samples // self.batch_size)

        print('Number of raw_tokens: ', sum([len(tok) for tok in self.train_tokens]))
        print('Number of samples in a batch: ', self.num_samples)
        print('Number of batches: ', self.num_batches)
    
    def random_choice(self, l):
        return l[random.randint(0, len(l)-1)]
    
    def category_tensor(self, category):
        li = self.all_categories.index(category)
        tensor = torch.tensor(build_vocab.get_vocab_indx_vector(self.vocab, self.spacy, "<" + category + ">")).to(device)
        return tensor, li
    
    def get_eval_item(self):
        return self.eval_tokens, self.eval_num_sequences

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Select property category/sequence by checking where sequence resides
        category_index = -1
        sequence_index = index
        while index >= 0:
            category_index += 1
            sequence_index = index
            index = index - self.train_num_sequences[category_index]

        string_category= self.all_categories[category_index]
        category, category_index = self.category_tensor(string_category)

        # Pick the right token samples based on the category
        current_sample = self.train_tokens[category_index]
            
        sequence_index = sequence_index * self.sequence_length
        end_index = sequence_index + self.sequence_length

        #print('seq index: ', sequence_index)
        #print('end index: ', end_index)
        return ( # might break if it gets the very end?
            torch.tensor(current_sample[sequence_index:end_index]).to(device), # x
            torch.tensor(current_sample[sequence_index+1:end_index+1]).to(device), # y
            category
        )

def train(model: nn.Module,
          dataset: torch.utils.data.Dataset,
          batch_size: int,
          sequence_length: int,
          num_epochs: int,
          ntokens: int,
          lr: float,
          type=0) -> None:
    model.train()  # turn on train mode
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) # Could try adam
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    total_loss = 0.
    log_interval = dataset.num_batches-1
    start_time = time.time()

    # Simple switch to deal with concatenation types for testing
    if type == 0:
        src_mask = generate_square_subsequent_mask(sequence_length).to(device)
    else:
        src_mask = generate_square_subsequent_mask(sequence_length+1).to(device)

    num_batches = dataset.num_batches

    for epoch in range(num_epochs):

        #for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        for batch, (x, y, category) in enumerate(dataloader):
            model.train()

            output = model(x, category, src_mask)

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
            
                wandb.log({"train_loss":cur_loss})
                wandb.log({"ppl":ppl})

                total_loss = 0
                start_time = time.time()
            if batch % num_batches-1 == 0 and batch > 0:
                eval_loss = evaluate(model, dataset, src_mask)
                print('EVAL LOSS: ', eval_loss)
                wandb.log({"eval_loss":eval_loss})

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def evaluate(model: nn.Module, dataset, src_mask) -> float:
    model.eval()  # turn on evaluation mode

    criterion = nn.CrossEntropyLoss()

    # Get eval data
    eval_tokens, eval_num_sequences = dataset.get_eval_item()

    total_loss = 0.

    with torch.no_grad():
        for index in range(0, sum(eval_num_sequences)):

            category_index = -1
            sequence_index = index
            while index >= 0:
                category_index += 1
                sequence_index = index
                index = index - eval_num_sequences[category_index]

            string_category= dataset.all_categories[category_index]
            category, category_index = dataset.category_tensor(string_category)

            # Pick the right token samples based on the category
            current_sample = eval_tokens[category_index]
                
            end_index = sequence_index + dataset.sequence_length
            
            data = torch.tensor(current_sample[sequence_index:end_index]).unsqueeze(0).to(device) # x
            targets = torch.tensor(current_sample[sequence_index+1:end_index+1]).unsqueeze(0).to(device) # y

            output = model(data, category.unsqueeze(0), src_mask)
            output_flat = output.transpose(1, 2)
            total_loss += criterion(output_flat, targets).item()
    return total_loss / (sum(eval_num_sequences) - 1)

def predict(model: nn.Module, input: str, category: str, dataset, generation_length=100):

    # Format the input
    prediction = build_vocab.get_vocab_indx_vector(dataset.vocab, build_vocab.load_spacy(), input)

    # Tokenize input
    sequence = torch.tensor(prediction).to(device)

    # Format tensor to be in shape: batch_size, seq_len
    sequence = sequence.reshape(1, -1)

    category = dataset.category_tensor(category)[0].reshape(1, -1)

    for _ in range(generation_length):
        # Give transformer entire sequence
        output = model(sequence, category)

        # Take last generated sequence of highest probability
        log_prob_vals, next_words = torch.topk(output[0,-1], k=10, dim=0)
        next_word = random.choice(next_words.tolist())

        # append to sequence
        prediction.append(next_word)
        sequence = torch.tensor(prediction).to(device).reshape(1, -1)

    # Decode entire sequence
    final_prediction = build_vocab.decode_vocab(dataset.vocab, prediction)

    return final_prediction

def predict_wrapper(model, dataset):

    input = 'in my younger and more vulnerable years' 
    category = 'greatgatsby'
    prediction = predict(model, input, category, dataset)

    print(' '.join(prediction))

    input = 'from fairest creatures' 
    category = 'shakespeare'
    prediction = predict(model, input, category, dataset)

    print(' '.join(prediction))

    input = 'tom was evidently perturbed at daisy’s running' 
    category = 'greatgatsby'
    prediction = predict(model, input, category, dataset)

    print(' '.join(prediction))

    input = 'in faith i do not love' 
    category = 'shakespeare'
    prediction = predict(model, input, category, dataset)

    print(' '.join(prediction))


# This is a way to perform multiple runs in the same script.
def train_wrapper():
    
    # Create datasets
    sequence_length = 256 # Length of one sequence
    batch_size = 16 # Number of sequences in a batch

    tag_type_books = 'books'
    books_dataset = Transformer_Dataset(sequence_length, batch_size, tag_type_books)

    tag_type_reviews = 'reviews'
    reviews_dataset = Transformer_Dataset(sequence_length, batch_size, tag_type_reviews)

    tag_type_scripts = 'scripts'
    scripts_dataset = Transformer_Dataset(sequence_length, batch_size, tag_type_scripts)

    ntokens_books = books_dataset.uniq_words  # size of vocabulary
    ntokens_reviews = reviews_dataset.uniq_words  # size of vocabulary
    print('ntokens_reviews: ', ntokens_reviews)
    ntokens_scripts = scripts_dataset.uniq_words  # size of vocabulary
    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    lrs = [0.0001, 0.001, 0.01]  # learning rates
    #num_epochs = 1
    
    # Normal
    print('---------------------')
    print('NORMAL')
    
    # for each learning rate
    for lr in lrs:

        #BOOKS

        run = wandb.init(name='normal',
                        project='controllable_transformer',
                        config={
                            'dataset':tag_type_books,
                            'epochs':500,
                            'hidden_size':d_hid,
                            'learning rate':lr
                        },
                        reinit=True
                        )
        
        model = transformer_model_category.TransformerModel_with_Category(ntokens_books, emsize, nhead, d_hid, nlayers, dropout).to(device)

        train(model, books_dataset, batch_size, sequence_length, 500, ntokens_books, lr, type=1)

        file_path = f"./trained_models/transformer_trained_normal_"+tag_type_books+"_"+str(lr)+".pt"

        torch.save(model.state_dict(), file_path)

        #predict_wrapper(model, books_dataset)

        run.finish()
        
        #REVIEWS

        run = wandb.init(name='normal',
                        project='controllable_transformer',
                        config={
                            'dataset':tag_type_reviews,
                            'epochs':10,
                            'hidden_size':d_hid,
                            'learning rate':lr
                        },
                        reinit=True
                        )
        
        model = transformer_model_category.TransformerModel_with_Category(ntokens_reviews, emsize, nhead, d_hid, nlayers, dropout).to(device)

        train(model, reviews_dataset, batch_size, sequence_length, 10, ntokens_reviews, lr, type=1)

        file_path = f"./trained_models/transformer_trained_normal_"+tag_type_reviews+"_"+str(lr)+".pt"

        torch.save(model.state_dict(), file_path)

        #predict_wrapper(model, reviews_dataset)

        run.finish()
        
        # SCRIPTS

        run = wandb.init(name='normal',
                        project='controllable_transformer',
                        config={
                            'dataset':tag_type_scripts,
                            'epochs':500,
                            'hidden_size':d_hid,
                            'learning rate':lr
                        },
                        reinit=True
                        )
        
        model = transformer_model_category.TransformerModel_with_Category(ntokens_scripts, emsize, nhead, d_hid, nlayers, dropout).to(device)

        train(model, scripts_dataset, batch_size, sequence_length, 500, ntokens_scripts, lr, type=1)

        file_path = f"./trained_models/transformer_trained_normal_"+tag_type_scripts+"_"+str(lr)+".pt"

        torch.save(model.state_dict(), file_path)

        #predict_wrapper(model, scripts_dataset)

        run.finish()

    """# Edited 1
    print('---------------------')
    print('EDITED 1')

    run = wandb.init(name='edited_1',
                     project='controllable_transformer',
                     config={
                        'dataset':tag_type,
                        'epochs':num_epochs,
                        'hidden_size':d_hid
                     },
                    reinit=True
                     )
    
    model = transformer_model_category_edited_1.TransformerModel_with_Category_edited(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    train(model, dataset, batch_size, sequence_length, num_epochs, ntokens, lr, type=0)

    file_path = f"./trained_models/transformer_trained_edited1.pt"

    torch.save(model.state_dict(), file_path)

    predict_wrapper(model, dataset)

    run.finish()

    # Edited 2
    print('---------------------')
    print('EDITED 2')

    run = wandb.init(name='edited_2',
                     project='controllable_transformer',
                     config={
                        'dataset':tag_type,
                        'epochs':num_epochs,
                        'hidden_size':d_hid
                     },
                    reinit=True
                     )
    
    model = transformer_model_category_edited_2.TransformerModel_with_Category_edited(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    train(model, dataset, batch_size, sequence_length, num_epochs, ntokens, lr, type=0)

    file_path = f"./trained_models/transformer_trained_edited2.pt"

    torch.save(model.state_dict(), file_path)

    predict_wrapper(model, dataset)

    run.finish()"""

    # Edited 3
    print('---------------------')
    print('EDITED 3')

    # for each learning rate
    for lr in lrs:

        #BOOKS

        run = wandb.init(name='edited_3',
                        project='controllable_transformer',
                        config={
                            'dataset':tag_type_books,
                            'epochs':500,
                            'hidden_size':d_hid,
                            'learning rate':lr
                        },
                        reinit=True
                        )
        
        model = transformer_model_category_edited_3.TransformerModel_with_Category_edited(ntokens_books, emsize, nhead, d_hid, nlayers, dropout).to(device)

        train(model, books_dataset, batch_size, sequence_length, 500, ntokens_books, lr, type=0)

        file_path = f"./trained_models/transformer_trained_edited_3_"+tag_type_books+"_"+str(lr)+".pt"

        torch.save(model.state_dict(), file_path)

        #predict_wrapper(model, books_dataset)

        run.finish()
        
        #REVIEWS

        run = wandb.init(name='edited_3',
                        project='controllable_transformer',
                        config={
                            'dataset':tag_type_reviews,
                            'epochs':10,
                            'hidden_size':d_hid,
                            'learning rate':lr
                        },
                        reinit=True
                        )
        
        model = transformer_model_category_edited_3.TransformerModel_with_Category_edited(ntokens_reviews, emsize, nhead, d_hid, nlayers, dropout).to(device)

        train(model, reviews_dataset, batch_size, sequence_length, 10, ntokens_reviews, lr, type=0)

        file_path = f"./trained_models/transformer_trained_edited_3_"+tag_type_reviews+"_"+str(lr)+".pt"

        torch.save(model.state_dict(), file_path)

        #predict_wrapper(model, reviews_dataset)

        run.finish()
        
        #SCRIPTS

        run = wandb.init(name='edited_3',
                        project='controllable_transformer',
                        config={
                            'dataset':tag_type_scripts,
                            'epochs':500,
                            'hidden_size':d_hid,
                            'learning rate':lr
                        },
                        reinit=True
                        )
        
        model = transformer_model_category_edited_3.TransformerModel_with_Category_edited(ntokens_scripts, emsize, nhead, d_hid, nlayers, dropout).to(device)

        train(model, scripts_dataset, batch_size, sequence_length, 500, ntokens_scripts, lr, type=0)

        file_path = f"./trained_models/transformer_trained_edited_3_"+tag_type_scripts+"_"+str(lr)+".pt"

        torch.save(model.state_dict(), file_path)

        #predict_wrapper(model, scripts_dataset)

        run.finish()

    # Edited 3.2
    print('---------------------')
    print('EDITED 3.2')

    # for each learning rate
    for lr in lrs:

        """BOOKS"""

        run = wandb.init(name='edited_3_2',
                        project='controllable_transformer',
                        config={
                            'dataset':tag_type_books,
                            'epochs':500,
                            'hidden_size':d_hid,
                            'learning rate':lr
                        },
                        reinit=True
                        )
        
        model = transformer_model_category_edited_3_2.TransformerModel_with_Category_edited(ntokens_books, emsize, nhead, d_hid, nlayers, dropout).to(device)

        train(model, books_dataset, batch_size, sequence_length, 500, ntokens_books, lr, type=0)

        file_path = f"./trained_models/transformer_trained_edited_3_2_"+tag_type_books+"_"+str(lr)+".pt"

        torch.save(model.state_dict(), file_path)

        #predict_wrapper(model, books_dataset)

        run.finish()
        
        """REVIEWS"""

        run = wandb.init(name='edited_3_2',
                        project='controllable_transformer',
                        config={
                            'dataset':tag_type_reviews,
                            'epochs':10,
                            'hidden_size':d_hid,
                            'learning rate':lr
                        },
                        reinit=True
                        )
        
        model = transformer_model_category_edited_3_2.TransformerModel_with_Category_edited(ntokens_reviews, emsize, nhead, d_hid, nlayers, dropout).to(device)

        train(model, reviews_dataset, batch_size, sequence_length, 10, ntokens_reviews, lr, type=0)

        file_path = f"./trained_models/transformer_trained_edited_3_2_"+tag_type_reviews+"_"+str(lr)+".pt"

        torch.save(model.state_dict(), file_path)

        #predict_wrapper(model, reviews_dataset)

        run.finish()
        
        """SCRIPTS"""

        run = wandb.init(name='edited_3_2',
                        project='controllable_transformer',
                        config={
                            'dataset':tag_type_scripts,
                            'epochs':500,
                            'hidden_size':d_hid,
                            'learning rate':lr
                        },
                        reinit=True
                        )
        
        model = transformer_model_category_edited_3_2.TransformerModel_with_Category_edited(ntokens_scripts, emsize, nhead, d_hid, nlayers, dropout).to(device)

        train(model, scripts_dataset, batch_size, sequence_length, 500, ntokens_scripts, lr, type=0)

        file_path = f"./trained_models/transformer_trained_edited_3_2_"+tag_type_scripts+"_"+str(lr)+".pt"

        torch.save(model.state_dict(), file_path)

        #predict_wrapper(model, scripts_dataset)

        run.finish()

    """# Edited 4
    print('---------------------')
    print('EDITED 4')

    run = wandb.init(name='edited_4',
                     project='controllable_transformer',
                     config={
                        'dataset':tag_type,
                        'epochs':num_epochs,
                        'hidden_size':d_hid
                     },
                    reinit=True
                     )
    
    model = transformer_model_category_edited_4.TransformerModel_with_Category_edited(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    train(model, dataset, batch_size, sequence_length, num_epochs, ntokens, lr, type=0)

    file_path = f"./trained_models/transformer_trained_edited4.pt"

    torch.save(model.state_dict(), file_path)

    predict_wrapper(model, dataset)

    run.finish()"""


def main():
    wandb.login()
    train_wrapper()


if __name__ == "__main__":
    main()
