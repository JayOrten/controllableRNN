#!/usr/bin/env python3

import pickle as pkl
import spacy
import re
import torchtext.vocab
from torchtext.vocab import build_vocab_from_iterator
from os.path import exists
import os
import torch
import sys

def tokenize(text, tokenizer):
    x = 0
    return [tok.text for tok in tokenizer.tokenizer(text)]

def yield_tokens(data_iter, tokenizer):
    for line in data_iter:
        yield tokenizer(line)

def build_vocabulary_from_file(spacy_en, filename: str, lowercase=True):
    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print(f"Building English Vocabulary from {filename} ...")
    # train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    with open(filename, encoding="utf-8") as f:
        if lowercase:
            train = f.read().lower().splitlines()
        else:
            train = f.read().splitlines()
    vocab = build_vocab_from_iterator(
        yield_tokens(train, tokenize_en),
        min_freq=1,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab.set_default_index(vocab["<unk>"])

    return vocab

def load_vocab(filename: str):
    if not exists(filename):
        # vocab = build_vocabulary_from_file(nlp, filename)
        # torch.save((vocab), filename)
        # From now on, I should use one function to build and save a new vocab, and another to load it. This is the load function
        raise Exception
    else:
        vocab = torch.load(filename)
    print("Finished.\nVocabulary sizes:")
    print(len(vocab))
    return vocab

def build_and_save_vocab_from_file(nlp, txt_filename, vocab_save_filename):
    vocab = build_vocabulary_from_file(nlp, txt_filename)
    torch.save((vocab), vocab_save_filename)
    return vocab

# This is how you get the initial vector form of the text. It's not exactly a one-hot
# vector, but it's similar. It's a vector of token indicies based on the vocab. (if
# 'I' is at position 23 in the vocab, it's vector form would be [23]
def get_vocab_indx_vector(vocab, tokenizer, text):
    return vocab([tok.text for tok in tokenizer.tokenizer(text)])

def decode_vocab(vocab: torchtext.vocab.Vocab, tokens_ind):
    return vocab.lookup_tokens(tokens_ind)

def generate_tokenized_file(vocab, tokenizer,  txt_filename: str, tok_filename: str, lowercase=False):
    with open(txt_filename, mode="r", encoding="utf-8") as txt_f:
        if lowercase:
            complete_txt = " ".join(filter(None, txt_f.read().lower().splitlines()))
        else:
            complete_txt = " ".join(filter(None, txt_f.read().splitlines()))
        with open(tok_filename, mode="wb") as tok_f:
            tokenized_txt = get_vocab_indx_vector(vocab, tokenizer, complete_txt)
            pkl.dump(tokenized_txt, tok_f)

def generate_str_tokenized_file(vocab, tokenizer,  txt_filename: str, tok_filename: str, lowercase=False):
    with open(txt_filename, mode="r", encoding="utf-8") as txt_f:
        if lowercase:
            complete_txt = " ".join(filter(None, txt_f.read().lower().splitlines()))
        else:
            complete_txt = " ".join(filter(None, txt_f.read().splitlines()))
        with open(tok_filename, mode="wb") as tok_f:
            tokenized_txt = get_vocab_indx_vector(vocab, tokenizer, complete_txt)
            tokenized_txt_str = [str(x) for x in tokenized_txt]

            pkl.dump(tokenized_txt_str, tok_f)

def load_tokenized_file(tok_filename):
    with open(tok_filename, "rb") as f:
        return pkl.load(f)
    
def main():
    # Pass in arguments like : ../data/reviews/music_small
    n = len(sys.argv)
    print('arguments: ', sys.argv)
    with open("combined.txt", "w") as outfile:
        for i in range(1, n):
            filename = sys.argv[i]
            print('filename: ', filename)
            with open(filename) as infile:
                contents = infile.read()
                outfile.write(contents)

    nlp = spacy.load("en_core_web_sm")
    # In order to create the vocab, you have to combine all of the sources
    vocab = build_and_save_vocab_from_file(nlp, "combined.txt", "mar_2023_lowercase_reviews_small_vocab.pt")

    specifiers = []
    # Get the token file tags:
    for i in range(1, n):
        filename = sys.arv[i]
        specifiers.append(filename.split('\\')[-1])

    for i in range(1, n):
        filename = sys.arv[i]
        output_file = "lowercase_" + specifiers[i] + "_tok.pkl"
        generate_tokenized_file(vocab, nlp, filename, output_file, lowercase=True)

    # Remove combined
    os.remove("combined.txt")


if __name__ == "__main__":
    main()
