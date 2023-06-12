import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

datapath = f'train.csv'
df = pd.read_csv(datapath)
df.head()

df.drop(columns = ["keyword", "id", "location"], inplace = True)
#df.groupby(['category']).size().plot.bar()

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [x for x in df['target']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 50, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], np.array(self.labels[idx])  
    
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        print(pooled_output)
        print(pooled_output.shape)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

def train(model, train_data, val_data, learning_rate, epochs, device_):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=1)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    model = model.to(device_)
    criterion = criterion.to(device_)

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device_)
                mask = train_input['attention_mask'].to(device_)
                input_id = train_input['input_ids'].squeeze(1).to(device_)

                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device_)
                    mask = val_input['attention_mask'].to(device_)
                    input_id = val_input['input_ids'].squeeze(1).to(device_)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            print(len(train_data))
            print(len(val_data))
            print(
                f'Epochs: {epoch_num + 1}\
                    | Train Loss: {total_loss_train / len(train_data): .3f}\
                    | Train Accuracy: {total_acc_train / len(train_data): .3f}\
                    | Val Loss: {total_loss_val / len(val_data): .3f}\
                    | Val Accuracy: {total_acc_val / len(val_data): .3f}')


def evaluate(model, test_data, device_):

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

    model = model.to(device_)

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device_)
              mask = test_input['attention_mask'].to(device_)
              input_id = test_input['input_ids'].squeeze(1).to(device_)

              output = model(input_id, mask)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    

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


def main():
    np.random.seed(112)
    sample = df.sample(frac=0.2, random_state=42)
    df_train, df_val, df_test = np.split(sample, 
                                        [int(.8*len(sample)), int(.9*len(sample))])
    print(len(df_train),len(df_val), len(df_test))
    device=set_up()

    #Hyperparamiters
    EPOCHS = 10
    model = BertClassifier()
    LR = 1e-6
        
    train(model, df_train, df_val, LR, EPOCHS, device)
    evaluate(model, df_test, device)

main()

                 