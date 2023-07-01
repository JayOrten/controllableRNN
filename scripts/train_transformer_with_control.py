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
from sacrebleu.metrics import BLEU
import nltk
from nltk import tokenize
from bert_score import BERTScorer

import build_vocab
import transformer_model_category
#import transformer_model_category_edited_1
#import transformer_model_category_edited_2
import transformer_model_category_edited_3
#import transformer_model_category_edited_3_2
#import transformer_model_category_edited_4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

nltk.download('punkt') # tokenizer used for evaluation metrics

def find_files(path): return glob.glob(path)

class Transformer_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sequence_length,
        batch_size,
        tag_type
    ):
        folder = "..\\vocabs_and_tokens\\" + tag_type + "\\"
        data_folder = "..\\data\\" + tag_type + "\\"
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

        self.raw_text = {} # used for metrics like BLEU. Dict of category -> [sentences in data]
        all_categories = []
        for filename in find_files(data_folder + '*.txt'):
            category = os.path.splitext(os.path.basename(filename))[0]
            all_categories.append(category)

            with open(filename, mode="r", encoding="utf-8") as txt_file:
                contents = txt_file.read()
                sentences = tokenize.sent_tokenize(contents) # Tokenizes into sentences
                self.raw_text[category] = sentences
        
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

        self.eval_tokens = [] # List of lists, corresponding to tokens from each dataset
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
    log_interval = dataset.num_batches-1 # Logging every epoch
    start_time = time.time()
    text_table = wandb.Table(columns=["epoch", "category", "generation"])

    # Simple switch to deal with concatenation types for testing
    if type == 0:
        src_mask = generate_square_subsequent_mask(sequence_length).to(device)
    else:
        src_mask = generate_square_subsequent_mask(sequence_length+1).to(device)

    num_batches = dataset.num_batches

    for epoch in range(num_epochs):
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

                predict_wrapper(model, dataset, epoch, text_table)

        scheduler.step()

    wandb.log({"training_samples" : text_table})

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def evaluate(model: nn.Module, dataset, src_mask) -> float:
    model.eval()  # turn on evaluation mode

    criterion = nn.CrossEntropyLoss()

    # Get eval data
    eval_tokens, eval_num_sequences = dataset.get_eval_item()

    total_loss = 0.

    # Run through test data and calulcate evaluation loss
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

            sequence_index = sequence_index * dataset.sequence_length
                
            end_index = sequence_index + dataset.sequence_length
            
            data = torch.tensor(current_sample[sequence_index:end_index]).unsqueeze(0).to(device) # x
            targets = torch.tensor(current_sample[sequence_index+1:end_index+1]).unsqueeze(0).to(device) # y

            output = model(data, category.unsqueeze(0), src_mask)
            output_flat = output.transpose(1, 2)
            total_loss += criterion(output_flat, targets).item()

    evaluation_loss = total_loss / (sum(eval_num_sequences) - 1)

    # Calculate BLEU score on an arbitrary set of 20 samples
    # Here we are getting a prediction and using the entire eval set as reference. We average of the ten samples.
    num_samples = 20
    sum_score = 0
    for _ in range(num_samples):
        # Pick random category
        category = random.choice(dataset.all_categories)

        # Get all sentences of data
        data = dataset.raw_text[category]

        # Get prediction
        input = "i"
        prediction = ' '.join(predict(model, input, category, dataset, generation_length=15))

        bleu = BLEU()

        bleu_score = bleu.corpus_score(prediction, data)

        sum_score += bleu_score.score

    # Average score of generations
    final_bleu_score = sum_score / num_samples

    # Record in WandB
    wandb.log({"bleu_score_corpus":final_bleu_score})

    # Here we are going through each sentence in corpus and generating a result based on half of the sentence.
    # Then, we do a corpus wide bleu score calculation on all generation. Seems like this should take a lot longer...
    # this doesn't work, it just takes way too long
    """references = []
    predictions = []
    # For each sentence in corpus
    for category in dataset.all_categories:
        for sentence in dataset.raw_text[category]:
            print('sentence: ', sentence)
            # Get length of sentence
            sentence_length = len(sentence.split())
            #print('sentence_length: ', sentence_length)

            # Get first half of sentence
            # If the sentence length < 2, just skip??
            if sentence_length >= 2:
                sentence_half = ' '.join(sentence.split()[:sentence_length//2])
                generation_length = sentence_length//2
            else:
                sentence_half = sentence
                generation_length = 1
            #print('sentence_half: ', sentence_half)

            # Generate prediction on the rest, up to the proper length
            prediction = ' '.join(predict(model, sentence_half.lower(), category, dataset, generation_length=generation_length))
            print('prediction: ', prediction)

            # Add full sentences for both reference and prediction to arrays
            references.append(sentence)
            predictions.append(prediction)
    
    # Compute full bleu score for all.
    bleu = BLEU()

    bleu_score = bleu.corpus_score(predictions, [references])
    print('bleu score: ', bleu_score)"""

    # Instead we do halfway prediction on a sample of 20 sentences or so from the set
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    references = []
    predictions = []

    num_samples = 20
    for _ in range(num_samples):
        # Pick random category
        category = random.choice(dataset.all_categories)

        # Get all sentences of data
        data = dataset.raw_text[category]

        # Get random sentence
        sentence = data[random.randint(0, len(data)-1)]

        # Get length of sentence
        sentence_length = len(sentence.split())

        # Get first half of sentence
        # If the sentence length < 2, just use the word
        if sentence_length >= 2:
            sentence_half = ' '.join(sentence.split()[:sentence_length//2])
            generation_length = sentence_length//2
        else:
            sentence_half = sentence
            generation_length = 1

        # Generate prediction on the rest, up to the proper length
        prediction = ' '.join(predict(model, sentence_half.lower(), category, dataset, generation_length=generation_length))
        
        # Add full sentences for both reference and prediction to arrays
        references.append(sentence)
        predictions.append(prediction)

    bleu = BLEU()

    bleu_score = bleu.corpus_score(predictions, [references])

    # Record in WandB
    wandb.log({"bleu_score":bleu_score.score})

    # BERTScore
    P, R, F1 = scorer.score(predictions, references)
    print('p: ', P)
    print('R: ', R)
    print('F1: ', F1)
    print(f"System level F1 score: {F1.mean()}")

    wandb.log({"system_level_f1_BERT":F1.mean()})

    #CTRLEval?

    return evaluation_loss

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

def predict_wrapper(model, dataset, epoch, text_table = None):

    categories = dataset.all_categories

    for category in categories:
        #print('category: ', category)

        input = 'i' 
        prediction = predict(model, input, category, dataset)

        #print(' '.join(prediction))

        with open('predictions.txt', 'a') as file:
            file.write('\n')
            file.write('Category: ')
            file.write(category)
            file.write('Prediction: ')
            file.write(' '.join(prediction))

        if text_table:
            text_table.add_data(epoch, category, ' '.join(prediction))


# This is a way to perform multiple runs in the same script.
def train_wrapper():
    
    # Create datasets
    sequence_length = 256 # Length of one sequence
    batch_size = 16 # Number of sequences in a batch

    tag_type_books_2_sources = 'books_2_sources'
    books_2_dataset = Transformer_Dataset(sequence_length, batch_size, tag_type_books_2_sources)

    tag_type_books_3_sources = 'books_3_sources'
    books_3_dataset = Transformer_Dataset(sequence_length, batch_size, tag_type_books_3_sources)

    tag_type_books_6_sources = 'books_6_sources'
    books_6_dataset = Transformer_Dataset(sequence_length, batch_size, tag_type_books_6_sources)

    tag_type_reviews = 'reviews'
    reviews_dataset = Transformer_Dataset(sequence_length, batch_size, tag_type_reviews)

    tag_type_scripts = 'scripts'
    scripts_dataset = Transformer_Dataset(sequence_length, batch_size, tag_type_scripts)

    ntokens_books_2 = books_2_dataset.uniq_words  # size of vocabulary
    ntokens_books_3 = books_3_dataset.uniq_words  # size of vocabulary
    ntokens_books_6 = books_6_dataset.uniq_words  # size of vocabulary
    ntokens_reviews = reviews_dataset.uniq_words  # size of vocabulary
    ntokens_scripts = scripts_dataset.uniq_words  # size of vocabulary

    emsize = 200  # embedding dimension
    d_hids = [128,256,512,1024]  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayerss = [2,4,6,8]  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    lr = 5.0  # learning rates
    num_epochs = 2
    project_name = "transformer_test"
    
    # Normal
    print('---------------------')
    print('NORMAL')
    
    # for each learning rate
    for d_hid in d_hids:
        for nlayers in nlayerss:

            #BOOKS 2 SOURCES
            run = wandb.init(name='normal_books_2_'+str(d_hid)+'_'+str(nlayers),
                            project=project_name,
                            config={
                                'dataset':tag_type_books_2_sources,
                                'epochs':num_epochs,
                                'hidden_size':d_hid,
                                'learning rate':lr,
                                'nlayers':nlayers
                            },
                            reinit=True
                            )
            
            model = transformer_model_category.TransformerModel_with_Category(ntokens_books_2, emsize, nhead, d_hid, nlayers, dropout).to(device)

            train(model, books_2_dataset, batch_size, sequence_length, num_epochs, ntokens_books_2, lr, type=1)

            file_path = f"./trained_models/transformer_trained_normal_"+tag_type_books_2_sources+"_"+str(d_hid)+"_"+str(nlayers)+".pt"

            torch.save(model.state_dict(), file_path)

            run.finish()
            
            #BOOKS 3 SOURCES

            run = wandb.init(name='normal_books_3_'+str(d_hid)+'_'+str(nlayers),
                            project=project_name,
                            config={
                                'dataset':tag_type_books_3_sources,
                                'epochs':num_epochs,
                                'hidden_size':d_hid,
                                'learning rate':lr,
                                'nlayers':nlayers
                            },
                            reinit=True
                            )
            
            model = transformer_model_category.TransformerModel_with_Category(ntokens_books_3, emsize, nhead, d_hid, nlayers, dropout).to(device)

            train(model, books_3_dataset, batch_size, sequence_length, num_epochs, ntokens_books_3, lr, type=1)

            file_path = f"./trained_models/transformer_trained_normal_"+tag_type_books_3_sources+"_"+str(d_hid)+"_"+str(nlayers)+".pt"

            torch.save(model.state_dict(), file_path)

            run.finish()

            #BOOKS 6 SOURCES 

            run = wandb.init(name='normal_books_6_'+str(d_hid)+'_'+str(nlayers),
                            project=project_name,
                            config={
                                'dataset':tag_type_books_6_sources,
                                'epochs':num_epochs,
                                'hidden_size':d_hid,
                                'learning rate':lr,
                                'nlayers':nlayers
                            },
                            reinit=True
                            )
            
            model = transformer_model_category.TransformerModel_with_Category(ntokens_books_6, emsize, nhead, d_hid, nlayers, dropout).to(device)

            train(model, books_6_dataset, batch_size, sequence_length, num_epochs, ntokens_books_6, lr, type=1)

            file_path = f"./trained_models/transformer_trained_normal_"+tag_type_books_6_sources+"_"+str(d_hid)+"_"+str(nlayers)+".pt"

            torch.save(model.state_dict(), file_path)

            run.finish()

            #REVIEWS

            run = wandb.init(name='normal_reviews_'+str(d_hid)+'_'+str(nlayers),
                            project=project_name,
                            config={
                                'dataset':tag_type_reviews,
                                'epochs':num_epochs,
                                'hidden_size':d_hid,
                                'learning rate':lr,
                                'nlayers':nlayers
                            },
                            reinit=True
                            )
            
            model = transformer_model_category.TransformerModel_with_Category(ntokens_reviews, emsize, nhead, d_hid, nlayers, dropout).to(device)

            train(model, reviews_dataset, batch_size, sequence_length, num_epochs, ntokens_reviews, lr, type=1)

            file_path = f"./trained_models/transformer_trained_normal_"+tag_type_reviews+"_"+str(d_hid)+"_"+str(nlayers)+".pt"

            torch.save(model.state_dict(), file_path)

            run.finish()
            
            # SCRIPTS

            run = wandb.init(name='normal_scripts_'+str(d_hid)+'_'+str(nlayers),
                            project=project_name,
                            config={
                                'dataset':tag_type_scripts,
                                'epochs':num_epochs,
                                'hidden_size':d_hid,
                                'learning rate':lr,
                                'nlayers':nlayers
                            },
                            reinit=True
                            )
            
            model = transformer_model_category.TransformerModel_with_Category(ntokens_scripts, emsize, nhead, d_hid, nlayers, dropout).to(device)

            train(model, scripts_dataset, batch_size, sequence_length, num_epochs, ntokens_scripts, lr, type=1)

            file_path = f"./trained_models/transformer_trained_normal_"+tag_type_scripts+"_"+str(d_hid)+"_"+str(nlayers)+".pt"

            torch.save(model.state_dict(), file_path)

            run.finish()

    # Edited 3
    print('---------------------')
    print('EDITED 3')

    # for each learning rate
    for d_hid in d_hids:
        for nlayers in nlayerss:

            #BOOKS 2 SOURCES

            run = wandb.init(name='edited_3_books_2_'+str(d_hid)+'_'+str(nlayers),
                            project=project_name,
                            config={
                                'dataset':tag_type_books_2_sources,
                                'epochs':num_epochs,
                                'hidden_size':d_hid,
                                'learning rate':lr,
                                'nlayers':nlayers
                            },
                            reinit=True
                            )
            
            model = transformer_model_category_edited_3.TransformerModel_with_Category_edited(ntokens_books_2, emsize, nhead, d_hid, nlayers, dropout).to(device)

            train(model, books_2_dataset, batch_size, sequence_length, num_epochs, ntokens_books_2, lr, type=0)

            file_path = f"./trained_models/transformer_trained_edited_3_"+tag_type_books_2_sources+"_"+str(d_hid)+"_"+str(nlayers)+".pt"

            torch.save(model.state_dict(), file_path)

            run.finish()
            
            #BOOKS 3 SOURCES

            run = wandb.init(name='edited_3_books_3_'+str(d_hid)+'_'+str(nlayers),
                            project=project_name,
                            config={
                                'dataset':tag_type_books_3_sources,
                                'epochs':num_epochs,
                                'hidden_size':d_hid,
                                'learning rate':lr,
                                'nlayers':nlayers
                            },
                            reinit=True
                            )
            
            model = transformer_model_category_edited_3.TransformerModel_with_Category_edited(ntokens_books_3, emsize, nhead, d_hid, nlayers, dropout).to(device)

            train(model, books_3_dataset, batch_size, sequence_length, num_epochs, ntokens_books_3, lr, type=0)

            file_path = f"./trained_models/transformer_trained_edited_3_"+tag_type_books_3_sources+"_"+str(d_hid)+"_"+str(nlayers)+".pt"

            torch.save(model.state_dict(), file_path)

            run.finish()

            #BOOKS 6 SOURCES

            run = wandb.init(name='edited_3_books_6_'+str(d_hid)+'_'+str(nlayers),
                            project=project_name,
                            config={
                                'dataset':tag_type_books_6_sources,
                                'epochs':num_epochs,
                                'hidden_size':d_hid,
                                'learning rate':lr,
                                'nlayers':nlayers
                            },
                            reinit=True
                            )
            
            model = transformer_model_category_edited_3.TransformerModel_with_Category_edited(ntokens_books_6, emsize, nhead, d_hid, nlayers, dropout).to(device)

            train(model, books_6_dataset, batch_size, sequence_length, num_epochs, ntokens_books_6, lr, type=0)

            file_path = f"./trained_models/transformer_trained_edited_3_"+tag_type_books_6_sources+"_"+str(d_hid)+"_"+str(nlayers)+".pt"

            torch.save(model.state_dict(), file_path)

            run.finish()

            #REVIEWS

            run = wandb.init(name='edited_3_reviews_'+str(d_hid)+'_'+str(nlayers),
                            project=project_name,
                            config={
                                'dataset':tag_type_reviews,
                                'epochs':num_epochs,
                                'hidden_size':d_hid,
                                'learning rate':lr,
                                'nlayers':nlayers
                            },
                            reinit=True
                            )
            
            model = transformer_model_category_edited_3.TransformerModel_with_Category_edited(ntokens_reviews, emsize, nhead, d_hid, nlayers, dropout).to(device)

            train(model, reviews_dataset, batch_size, sequence_length, num_epochs, ntokens_reviews, lr, type=0)

            file_path = f"./trained_models/transformer_trained_edited_3_"+tag_type_reviews+"_"+str(d_hid)+"_"+str(nlayers)+".pt"

            torch.save(model.state_dict(), file_path)

            run.finish()
            
            #SCRIPTS

            run = wandb.init(name='edited_3_scripts_'+str(d_hid)+'_'+str(nlayers),
                            project=project_name,
                            config={
                                'dataset':tag_type_scripts,
                                'epochs':num_epochs,
                                'hidden_size':d_hid,
                                'learning rate':lr,
                                'nlayers':nlayers
                            },
                            reinit=True
                            )
            
            model = transformer_model_category_edited_3.TransformerModel_with_Category_edited(ntokens_scripts, emsize, nhead, d_hid, nlayers, dropout).to(device)

            train(model, scripts_dataset, batch_size, sequence_length, num_epochs, ntokens_scripts, lr, type=0)

            file_path = f"./trained_models/transformer_trained_edited_3_"+tag_type_scripts+"_"+str(d_hid)+"_"+str(nlayers)+".pt"

            torch.save(model.state_dict(), file_path)

            run.finish()


def main():
    wandb.login()
    train_wrapper()


if __name__ == "__main__":
    main()
