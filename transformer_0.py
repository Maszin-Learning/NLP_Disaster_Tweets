import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.optim import Adam
from tqdm import tqdm



class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx)
        src_mask=src_mask.unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)


    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
    
    
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [x for x in df['target']]
        
        self.texts = [tokenizer(text, 
                    padding='max_length', max_length = 10, truncation=True,
                    return_tensors="pt") for text in df['text']]
        
        self.texts = [torch.cat([ torch.tensor([1]), i['input_ids'][0], torch.tensor([0])], dim=0) for i in self.texts]
        self.texts = [torch.unsqueeze(x,0)  for x in self.texts] 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

#define device
def set_up():
    if torch.cuda.is_available():
        device_ = torch.device("cuda")
        print (f"Using {device_}")
        #Checking GPU RAM allocated memory
        print('allocated CUDA memory: ',torch.cuda.memory_allocated())
        print('cached CUDA memory: ',torch.cuda.memory_cached())
        torch.cuda.empty_cache() # clear CUDA memory
        torch.backends.cudnn.benchmark = True #let cudnn chose the most efficient way of calculating Convolutions
        
    elif torch.backends.mps.is_available():
        print ("CUDA device not found.")
        device_ = torch.device("mps")
        print (f"Using {device_}")
    else:
        print ("MPS device not found.")
        device_ = torch.device("cpu")
        print (f"Using {device_}")
    return device_

#dodać learning loop
def train(model, net, train_data, test_data, learning_rate, epochs, device, batch_size):

    #device = set_up()
    train, test = Dataset(train_data), Dataset(test_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    print('passed setup')
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr = learning_rate)

    model = model.to(device)
    criterion = criterion.to(device)

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for input, output in tqdm(train_dataloader):

                # change device
                output = output.to(device)
                input = input.to(device)

                #print(output[0].shape)
                print(input.shape)
                input = input
                print(input.shape)
                print(input[:,:-1].shape)

                # throw input into model
                output = model(input, input[:, :-1])
                print('wors')
                #net_output=net(torch.flatten(output))
                batch_loss = criterion(net_output, output.long())

                # calculate accuracy
                total_loss_train += batch_loss.item()
                acc = (output.argmax(dim=1) == output).sum().item()
                total_acc_train += acc

                # calculate gradient and update weights
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_target in test_dataloader:
                    
                    #change device
                    val_input = val_input.to(device)
                    val_target = val_target.to(device)

                    # throw input into model
                    output = model(val_input, val_input[:, :-1])
                    batch_loss = criterion(output, val_target.long())

                    # calculate accuracy
                    total_loss_val += batch_loss.item()
                    acc = (output.argmax(dim=1) == val_target).sum().item()
                    total_acc_val += acc

            print(
                f'Epochs: {epoch_num + 1}\
                    | Train Loss: {total_loss_train / len(train_data): .3f}\
                    | Train Accuracy: {total_acc_train / len(train_data): .3f}\
                    | Val Loss: {total_loss_val / len(test): .3f}\
                    | Val Accuracy: {total_acc_val / len(test): .3f}')

# Network class
class NET(nn.Module):
    def __init__(self,input_size,output_size, dropout=0.5):
        # super function. It inherits from nn.Module and we can access everything in nn.Module
        super(NET,self).__init__()
        # Linear function.
        self.linear_0 = nn.Linear(input_size,1000)
        self.linear_1 = nn.Linear(1000, 1000)
        self.linear_2 = nn.Linear(1000,output_size)
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear_0(x)
        #x = self.dropout(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        final_layer = self.relu(x)
        return final_layer

#dodać eval loop
if __name__ == "__main__":
    
    #device = set_up()
    device = torch.device('cpu') #temporary use cpu
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    #data processing
    datapath = f'train.csv'
    df = pd.read_csv(datapath)
    df.head()
    df.drop(columns = ["keyword", "id", "location"], inplace = True)
    
    #data split
    np.random.seed(112)
    sample = df.sample(frac=0.2, random_state=42)
    df_train, df_val, df_test = np.split(sample, 
                                    [int(.8*len(sample)), int(.9*len(sample))])
    print(len(df_train),len(df_val), len(df_test))
    
    
    #data
    dataset = Dataset(df_train)
    x = dataset[3][0]
    y = dataset[3][1]
    
    print('x_shape',x.shape)
    print(x[:,:-1].shape)
    #print(y.shape)
    
    #hyperparameters
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 40000
    trg_vocab_size = 40000
    #dodac długośc tych słowników/zdań wszystkich itd
    train, test = Dataset(df_train), Dataset(df_test)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    #model
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device)
    net = NET(src_vocab_size*11,1)
    element = next(iter(train_dataloader))
    x_z = element[0]
    y_z = element[1]
    
    print('x_z shape',x_z.shape)
    
    #test 1
    try:
        out = model(x, x[:,:-1])
        print(torch.flatten(out).shape)
        print("PASSED TRANSFORMER TEST 1")
        net_output=net(torch.flatten(out))
        print('TEST 1 COMPLETED')
    except:
        print('TEST 1 FAILED')
        out = model(x, x[:,:-1])
        print("PASSED TRANSFORMER TEST 1")
        print(torch.flatten(out).shape)
        net_output=net(torch.flatten(out))
        
"""    
    #test 2
    try:
        out = model(x_z, x_z[:,:-1])
        print("PASSED TRANSFORMER TEST 2")
        print(torch.flatten(out).shape)
        net_output=net(torch.flatten(out))
        print('TEST 2 COMPLETED')
    except:
        print('TEST 2 FAILED')
        out = model(x_z, x_z[:,:-1])
        print("PASSED TRANSFORMER TEST 2")
        print(torch.flatten(out).shape)
        net_output=net(torch.flatten(out))
        
"""

    
    
train(model=model, net=net, train_data=df_train, test_data=df_test, learning_rate=1e-5, epochs=1, device=device, batch_size=1)



