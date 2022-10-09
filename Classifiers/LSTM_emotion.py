#from cgi import test
from cProfile import label
from matplotlib.pyplot import text
#import csv
import numpy as np
#import tensorflow as tf
#from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics import precision_recall_fscore_support
#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import TfidfVectorizer
#import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, vstack
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchmetrics.functional import f1_score
from torchmetrics.functional import precision
from torchmetrics.functional import recall
#from sklearn.preprocessing import LabelBinarizer


from datasets import load_dataset


#Daten laden aus huggingface
train = load_dataset('emotion', split='train')
test  = load_dataset('emotion', split='test')
sample = ['''A Woman's Story, A Man's Place, and Simple Passion were recognised as The New York Times Notable Books,[19] and A Woman's Story was a finalist for the Los Angeles Times Book Prize.[20] Shame was named a Publishers Weekly Best Book of 1998,[21] I Remain in Darkness a Top Memoir of 1999 by The Washington Post, and The Possession was listed as a Top Ten Book of 2008 by More magazine.[22]

Her 2008 historical memoir Les Années (The Years), well-received by French critics, is considered by many to be her magnum opus.[23] In this book, Ernaux writes about herself in the third person ('elle', or 'she' in English) for the first time, providing a vivid look at French society just after the Second World War until the early 2000s.[24] It is the story of a woman and of the evolving society she lived in. The Years won the 2008 Prix François-Mauriac de la région Aquitaine [fr],[25] the 2008 Marguerite Duras Prize,[26] the 2008 Prix de la langue française, the 2009 Télégramme Readers Prize, and the 2016 Strega European Prize. Translated by Alison L. Strayer, The Years was a finalist for the 31st Annual French-American Foundation Translation Prize, was nominated for the International Booker Prize in 2019,[27] and won the 2019 Warwick Prize for Women in Translation.[9][28] Her popularity in anglophone countries increased sharply after The Years was shortlisted for the International Booker.[29]

On 6 October 2022, it was announced that she was to be awarded the 2022 Nobel Prize in Literature[30][31] "for the courage and clinical acuity with which she uncovers the roots, estrangements and collective restraints of personal memory".[2] Ernaux is the 16th French writer, and the first Frenchwoman, to receive the literature prize.[30] In congratulating her, the president of France, Emmanuel Macron, said that she was the voice "of the freedom of women and of the forgotten".[30] ''']


#Umformatierung von dictionary in Listen
def convert(data_from_dict, split):
    list = []
    for i in range(len(data_from_dict)):
        list.append(data_from_dict[i][split])
    return list

train_text = convert(train, 'text')
train_label = convert(train, 'label')
test_text = convert(test, 'text')
test_label = convert(test, 'label')


def text_to_list_of_sentences(text_list):
    text_list = text_list[0].split(".")
    return text_list

sample = text_to_list_of_sentences(sample)


def pred_validation(pred):
    list = []
    for w in pred:
        sum = torch.sum(w).item()
        row = w.tolist()
        for x in range(len(row)):
           row[x] = round(row[x] / sum, 4)
        list.append(row)
    return list

#Daten zu one-hot wordvector machen
vec = CountVectorizer(ngram_range=(1,1), lowercase=True)
trn_x = vec.fit_transform(train_text)
feature_size = trn_x.shape[1]

def word_lengths_tolist(texts):
    word_lengths = []
    for text in texts:
        word_lengths.append(len(text.split()))

    return word_lengths

word_lengths = word_lengths_tolist(train_text)
maxlen = max(word_lengths)



def to_3d_sparse(texts, word_lenghts, maxlen):
    list_of_csr = []
    np_zero_vec = np.zeros(feature_size)
    csr_zero_vec = csr_matrix(np_zero_vec)
    for i in range(len(texts)):
        text_split = texts[i].split()
        word_vec = vec.transform(text_split)
        #np_word_vec = word_vec.todense()
        pad_size = maxlen - word_lenghts[i]
        for j in range(pad_size):
            word_vec = vstack((word_vec, np_zero_vec))
        
        #word_vec = csr_matrix(np_word_vec)
    print(word_vec.shape)
    print(i)
    return list_of_csr

data = to_3d_sparse(train_text, word_lengths, maxlen)
print(data.shape)






"""
def text_to_sparse(list_of_text):
    
    text_x = vec.transform(list_of_text)
    text_x_coo= coo_matrix(text_x)

    text_values = text_x_coo.data
    text_indices = np.vstack((text_x_coo.row, text_x_coo.col))

    text_i = torch.LongTensor(text_indices)
    text_v = torch.FloatTensor(text_values)
    text_shape = text_x_coo.shape

    #Making the test and train tensors for the text
    text_x_tensor = torch.sparse.FloatTensor(text_i, text_v, torch.Size(text_shape))
    return text_x_tensor

trn_x_tensor = text_to_sparse(train_text)
#tst_x_tensor = text_to_sparse(test_text)
tst_x_tensor = text_to_sparse(sample)
#Making y which is the tensor of emotion labels
y = torch.tensor(train_label)
y_test = torch.tensor(test_label)

#Setup CUDA if available, else CPU
print("cudNN Version", torch.backends.cudnn.version())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Creating Dataset
class SparseDataset(Dataset):
    def __init__(self, mat_csc, label, device="cpu"):
        self.dim = mat_csc.shape
        self.device = torch.device(device)

        csr = mat_csc.tocsr(copy=True)
        self.indptr = torch.tensor(csr.indptr, dtype=torch.int64, device=self.device)
        self.indices = torch.tensor(csr.indices, dtype=torch.int64, device=self.device)
        self.data = torch.tensor(csr.data, dtype=torch.float32, device=self.device)

        self.label = torch.tensor(label, dtype=torch.float32, device=self.device)

    def __len__(self):
        return self.dim[0]

    def __getitem__(self, idx):
        obs = torch.zeros((self.dim[1],), dtype=torch.float32, device=self.device)
        ind1,ind2 = self.indptr[idx],self.indptr[idx+1]
        obs[self.indices[ind1:ind2]] = self.data[ind1:ind2]

        return obs,self.label[idx]

train_ds = SparseDataset(trn_x, y)

# Define Dataloader and hyperparameters
hidden_size = 128
num_layers = 8
batch_size = 100 #batchsize depends on available memory
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Define model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN,self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.out_act = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        print(h0.shape)
        print(x.shape)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

#Initialize model
input_size = trn_x_tensor.shape[1]
output_size = 6

model = RNN(input_size, hidden_size, num_layers, output_size)


# Define loss function
loss_fn = nn.CrossEntropyLoss()
# Loss before training
#print("loss before training: ")


# Define optimizer
opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)


# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        # Train with batches of data
        for xb,yb in train_dl:
            yb = torch.tensor(yb, dtype=torch.long) # 0. setting right dtype for loss_fn (long required)
            pred = model(xb)                        # 1. Generate predictions
            loss = loss_fn(pred, yb)                # 2. Calculate loss
            loss.backward()                         # 3. Compute gradients
            opt.step()                              # 4. Update parameters using gradients
            opt.zero_grad()                         # 5. Reset the gradients to zero
        
        # Print the progress
        if (epoch+1) % 1 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# FIT THE MODEL
epochs = 700 #if choosing smaller batches go for less epochs
fit(epochs, model, loss_fn, opt, train_dl)
torch.save(model, "model.pth")
#model = torch.load("model700e_1e-4wd.pth")


# TEST THE MODEL
pred_test = model(tst_x_tensor)

pred_percentage = pred_validation(pred_test)
print(pred_percentage)

# Print results
p  = precision(pred_test, y_test, num_classes=6)
r  =    recall(pred_test, y_test, num_classes=6)
f1 =  f1_score(pred_test, y_test, num_classes=6)
print("F1-score:", f1)
print("Precision:", p)
print("Recall:", r)
"""

