import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRU_category(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(GRU_category, self).__init__()
        self.input_size = input_size # 4822
        self.hidden_size = hidden_size # 1400
        self.output_size = output_size # 4822
        self.n_layers = n_layers

        # EMBEDDING
        self.word_embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size)
        self.cat_embedding = nn.Embedding(num_embeddings=2, embedding_dim=hidden_size)

        self.gru = nn.GRU(hidden_size*2, hidden_size, n_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_token, hidden, batch_size, category):
        #print('input_token shape: ', input_token.size()) # 16, 256 -- batch_size, sequence_len
        #print('cat size: ', category.size()) # [16, 1]
        #print('cat: ', category)

        # To determine:
        # Do we concatenate the category and input token together, before embedding?
        # Do we embed them seperately in different spaces, and then concatenate?
        # Do we only embed the input, and then just concatenate the category?
        # Do we do addition, or concatenation?
        # Try embed both and then concatenate, but use the same embedding module.

        embedded_word = self.word_embedding(input_token)
        #print('embedded size: ', embedded_word.size()) # 16, 256, 1400 -- batch, sequence_length, input_size
        
        embedded_cat = self.cat_embedding(category)
        #print('embedded cat size: ', embedded_cat.size()) # [16, 1, 64]

        combined = torch.cat((embedded_word, embedded_cat), 2)

        #print('combined size: ', combined.size())

        out, hidden = self.gru(combined, hidden)
        
        out = self.fc(out)

        #print('out shape: ', out.size()) # 16, 256, 4822
        #print('hidden shape: ', hidden.size()) # 3, 16, 1400

        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device) # num_layers, batch_size, hidden_size
    

class GRU_with_cells_category(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(GRU_with_cells_category, self).__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # EMBEDDING
        self.word_embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size)
        self.cat_embedding = nn.Embedding(num_embeddings=2, embedding_dim=hidden_size)

        #self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=0.2, batch_first=True)
        # batch_size, input_size
        self.gru_cell_1 = nn.GRUCell(hidden_size*2, hidden_size) # dropout?

        self.gru_cell_2 = nn.GRUCell(hidden_size, hidden_size)

        self.gru_cell_3 = nn.GRUCell(hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_token, hidden_state, batch_size, category):
        hidden_1, hidden_2, hidden_3 = hidden_state # should unpack

        # input_token size: 16, 256

        embedded_word = self.word_embedding(input_token) # You'll have to check size on this
        #print('embedded size: ', embedded_word.size()) # 16, 256, 1400 -- batch, sequence_length, input_size
        
        embedded_cat = self.cat_embedding(category)
        #print('embedded cat size: ', embedded_cat.size()) # [16, 1, 64]

        combined = torch.cat((embedded_word, embedded_cat), 2)

        final_output_tensor = torch.empty((batch_size, input_token.size()[1], self.output_size)).to(device) # Final output needs to be 16, 256, 4822, or batch_size, sequence_length, output_size
        # Embedded size: 16, 256, 1400 -- batch, sequence_length, input_size
        for index in range(input_token.size()[1]):
            initial = combined[:,index,:]
            #print('initial size: ', initial.size()) # 16, 1400 prediction: 1, 256
            #print('hidden_size: ', hidden_1.size()) # 16, 1400 - 1, 256
            hidden_1 = self.gru_cell_1(initial, hidden_1)
            hidden_2 = self.gru_cell_2(hidden_1, hidden_2)
            hidden_3 = self.gru_cell_3(hidden_2, hidden_3)

            out = self.fc(hidden_3)

            #print('out size: ', torch.unsqueeze(out, 1).size()) # [16, 1, 4822] 1, 1, 12923
            # append to output tensor?
            final_output_tensor[:, index, :] = torch.unsqueeze(out, 1)[:, 0, :]

        #print("final_output_tensor size: ", final_output_tensor.size()) # 16, 256, 4822
        # Ultimately, we want to return the final out vector of all predictions
        return final_output_tensor, (hidden_1, hidden_2, hidden_3)

    def init_hidden(self, batch_size):
        hidden_1 = torch.zeros(batch_size, self.hidden_size).to(device) # num_layers, batch_size, hidden_size
        hidden_2 = torch.zeros(batch_size, self.hidden_size).to(device)
        hidden_3 = torch.zeros(batch_size, self.hidden_size).to(device)
        return (hidden_1, hidden_2, hidden_3)
    

class GRU_with_cells_category_edited(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(GRU_with_cells_category_edited, self).__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # EMBEDDING
        self.word_embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size)
        self.cat_embedding = nn.Embedding(num_embeddings=2, embedding_dim=hidden_size)

        #self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=0.2, batch_first=True)
        # batch_size, input_size
        self.gru_cell_1 = nn.GRUCell(hidden_size*2, hidden_size)

        self.gru_cell_2 = nn.GRUCell(hidden_size*2, hidden_size)

        self.gru_cell_3 = nn.GRUCell(hidden_size*2, hidden_size)

        self.fc = nn.Linear(hidden_size*2, output_size)
        
    def forward(self, input_token, hidden_state, batch_size, category):
        hidden_1, hidden_2, hidden_3 = hidden_state # should unpack

        # input_token size: 16, 256

        embedded_word = self.word_embedding(input_token) # You'll have to check size on this
        #print('embedded size: ', embedded_word.size()) # 16, 256, 1400 -- batch, sequence_length, input_size
        #print('category size: ', category.size())
        embedded_cat = self.cat_embedding(category)
        #print('embedded cat size: ', embedded_cat.size()) # [16, 1, 64]

        combined = torch.cat((embedded_word, embedded_cat), 2)

        final_output_tensor = torch.empty((batch_size, input_token.size()[1], self.output_size)).to(device) # Final output needs to be 16, 256, 4822, or batch_size, sequence_length, output_size
        # Embedded size: 16, 256, 1400 -- batch, sequence_length, input_size
        # input_token.size()[1] is the sequence length
        for index in range(input_token.size()[1]):
            initial = combined[:,index,:]
            cat = embedded_cat[:, index, :]
            #print('initial size: ', initial.size()) # 16, 1400
            #print('hidden_size: ', hidden_1.size()) # 16, 1400
            hidden_1 = self.gru_cell_1(initial, hidden_1)
            #print('hidden_1 size: ', hidden_1.size()) # 16, 256
            hidden_2 = self.gru_cell_2(torch.cat((cat,hidden_1), 1), hidden_2)
            hidden_3 = self.gru_cell_3(torch.cat((cat,hidden_2), 1), hidden_3)

            out = self.fc(torch.cat((cat,hidden_3),1))

            #print('out size: ', torch.unsqueeze(out, 1).size()) # [16, 1, 4822]
            # append to output tensor?
            final_output_tensor[:, index, :] = torch.unsqueeze(out, 1)[:, 0, :]

        #print("final_output_tensor size: ", final_output_tensor.size()) # 16, 256, 4822
        # Ultimately, we want to return the final out vector of all predictions
        return final_output_tensor, (hidden_1, hidden_2, hidden_3)

    def init_hidden(self, batch_size):
        hidden_1 = torch.zeros(batch_size, self.hidden_size).to(device) # num_layers, batch_size, hidden_size
        hidden_2 = torch.zeros(batch_size, self.hidden_size).to(device)
        hidden_3 = torch.zeros(batch_size, self.hidden_size).to(device)
        return (hidden_1, hidden_2, hidden_3)