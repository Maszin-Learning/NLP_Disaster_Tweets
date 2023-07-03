import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset, DataSet

# load data

test_data = pd.read_csv("test.csv")
train_data = pd.read_csv("train.csv")
samples = pd.read_csv("sample_submission.csv") # how data we send to Kaggle should look like

# train data consists out of 5 columns : "id", "keyword", "location", "text", "target", while test data has no "target"
# as we do not focus on reaching very high accuracy I'll delete "keyword" and "location"
# i will also delete "id" as we can index by row number

train_data.drop(columns = ["keyword", "id", "location"], inplace = True)
test_data.drop(columns = ["keyword", "id", "location"], inplace = True)

train_short = train_data.drop(train_data.index[100:], inplace = False)

# train data consist out of 7613 observations, including 4342 zeroes (false alarm) and 3271 ones (boooom)
# test data consist out of 3261 observation

# for simplicity let's reorganize it into arrays

train_text = train_data["text"]
train_target = train_data["target"]
test_text = test_data["text"]

# tokenize input to make it understandable for Bertie

model_name = 'bert-base-uncased' # smaller version of Bert NOT seeing any difference between small and big letters (B = b) or special signs (ó = o)

tokenizer = BertTokenizer.from_pretrained(model_name)

train_text_toc = []
test_text_toc = []
for sentence in train_text:
    train_text_toc.append(tokenizer(sentence, 
                                    padding = "max_length", # if we want to pad some "zeroes" to make all sentences of equal length
                                    max_length = 54,        # maximum length (in words) of sentence in train_data (I counted spaces)
                                    truncation = True,      # if some word is actually longer, than it shall be cut
                                    return_tensors = "pt")) # we want to get pytorch tensors
for sentence in test_text:
    test_text_toc.append(tokenizer(sentence, 
                                    padding = "max_length",
                                    max_length = 54,      
                                    truncation = True,     
                                    return_tensors = "pt"))
    
# NOTE: a tokenized sentenced consist out of 3 lists of tensors:
# 1. tokenized words + some "zeroes" to make the length constant
# 2. sentence number - we have here mostly just zeroes as we have usually just one sentence
# 3. info if a word is a real word or just padded "zero"

class myDataset(Dataset):
    def __init__(self, list_of_tokenized_text):
        self.labels = [0, 1]
        self.texts = list_of_tokenized_text

    def __getitem__(self, idx):
        return self.labels[idx]


'''

    
# and now load train data to a DataLoader

batch_size = 100

data = DataLoader(TensorDataset(train_text_toc, train_target), shuffle = True, batch_size = batch_size)
'''

input = tokenizer(train_text[0], 
                                    padding = "max_length", # if we want to pad some "zeroes" to make all sentences of equal length
                                    max_length = 54,        # maximum length (in words) of sentence in train_data (I counted spaces)
                                    truncation = True,      # if some word is actually longer, than it shall be cut
                                    return_tensors = "pt") # we want to get pytorch tensors

model = BertModel.from_pretrained(model_name)
output = model(**input)