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
test_word_lengths = word_lengths_tolist(test_text)
maxlen = max(word_lengths)
test_maxlen = max(test_word_lengths) 


# Tensor with dense dim(1) + sparse dim(2)
def to_3d_sparse_tensor(texts, word_lenghts, maxlen):
    list_of_tensors = []
    np_zero_vec = np.zeros(feature_size)
    csr_zero_vec = csr_matrix(np_zero_vec)
    for i in range(len(texts)):
        text_split = texts[i].split()
        word_vec = vec.transform(text_split)

         # Filling up with zeros up to maxlen
        pad_size = maxlen - word_lenghts[i]
        for j in range(pad_size):                  
            word_vec = vstack((word_vec, csr_zero_vec))
            
        #csr to sparse tensor
        vec_coo= coo_matrix(word_vec)

        vec_values = vec_coo.data
        vec_indices = np.vstack((vec_coo.row, vec_coo.col))

        vec_i = torch.LongTensor(vec_indices)
        vec_v = torch.FloatTensor(vec_values)
        vec_shape = vec_coo.shape
        vec_tensor = torch.sparse.FloatTensor(vec_i, vec_v, torch.Size(vec_shape))
        list_of_tensors.append(vec_tensor)


    data_tensor = torch.stack((list_of_tensors), 0)
    #Making the test and train tensors for the text
    
    return data_tensor

#Making and saving traiing data in 3d sparse tensor
try:
    train_data = torch.load('train_tensor.pt')
except:
    train_data = to_3d_sparse_tensor(train_text, word_lengths, maxlen)
    torch.save(train_data, 'train_tensor.pt')

test_data = to_3d_sparse_tensor(test_text, test_word_lengths, test_maxlen)

#Making y which is the tensor of emotion labels
y = torch.tensor(train_label)
y_test = torch.tensor(test_label)
print("Created training tensor succesfully", '\n', '\n')

#Setup CUDA if available, else CPU
print("cudNN Version", torch.backends.cudnn.version())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Creating Dataset
class SparseDataset(Dataset):
    def __init__(self, data, label, device=device):
        self.dim = data.shape
        self.device = torch.device(device)
        self.data = torch.tensor(data, dtype=torch.float64, device=self.device)
        
        self.label = torch.tensor(label, dtype=torch.float32, device=self.device)

    def __len__(self):
        return self.dim[0]

    def __getitem__(self, idx):
        current = self.data[idx].to_dense()
        return  current, self.label[idx]

train_ds = SparseDataset(train_data, y)
test_ds = SparseDataset(test_data, y_test)

# Define Dataloader and hyperparameters
hidden_size = 64
num_layers = 2
batch_size = 120 #batchsize depends on available memory
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

def outputfix(pred_tensor, num_layers, current_device=device):

    out = torch.zeros(6).to(current_device)
    for i in range(1, pred_tensor.shape[0]):
        idx = i
        if i % num_layers != 0:
            out = torch.vstack((out, pred_tensor[i-1:i:]))
    return out[1:]

# Define model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device=device):
        super(LSTM,self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, device=device)
        self.fc1 =  nn.Linear(hidden_size, 128, device=device) #fully connected 1
        self.fc = nn.Linear(128, output_size, device=device) #fully connected last layer
        
        self.relu = nn.ReLU()
    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.double, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.double, device=device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output

        return out

#Initialize model
input_size = train_data.shape[2]
output_size = 6

model = LSTM(input_size, hidden_size, num_layers, output_size).double().to(device=device)


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
        counter = 0
        for xb,yb in train_dl:
            
            yb = torch.tensor(yb, dtype=torch.long) # 0. setting right dtype for loss_fn (long required)
            pred = model(xb.double()) 
            #print(pred)              # 1. Generate predictions
            pred = outputfix(pred, num_layers)  
            #print(pred)                                       
            loss = loss_fn(pred, yb)                # 2. Calculate loss
            loss.backward()                         # 3. Compute gradients
            opt.step()                              # 4. Update parameters using gradients
            opt.zero_grad()                         # 5. Reset the gradients to zero
            counter = counter+1
            print('Batch {}/{} finished'.format(counter, len(train_dl)))
        # Print the progress
        if (epoch+1) % 1 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# FIT THE MODEL
epochs = 1 #if choosing smaller batches go for less epochs
fit(epochs, model, loss_fn, opt, train_dl)
torch.save(model, "model.pth")
#model = torch.load("model700e_1e-4wd.pth")

#Setting memory of GPU free
torch.cuda.empty_cache()
# TEST THE MODEL

batch_size = 100
test_ds = SparseDataset(test_data, y_test)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
#pred_test = outputfix(model(train_dl), num_layers)

def text_dl_to_predictions():
    test_preds = torch.zeros(6).to(device=device)
    for xb,_ in test_dl:
        test_batch = outputfix(model(xb.double()), num_layers, current_device=device)
        test_preds = torch.vstack((test_preds, test_batch))
    return test_preds[1:]

with torch.no_grad():
    pred_test = text_dl_to_predictions()
#pred_percentage = pred_validation(pred_test)

#print(pred_percentage)

# Print results
y_test = y_test.to(device=device)
p  = precision(pred_test, y_test, num_classes=6)
r  =    recall(pred_test, y_test, num_classes=6)
f1 =  f1_score(pred_test, y_test, num_classes=6)
print("F1-score:", f1)
print("Precision:", p)
print("Recall:", r)


