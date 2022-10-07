#from cgi import test
from cProfile import label
#import csv
import numpy as np
#import tensorflow as tf
#from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics import precision_recall_fscore_support
#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import TfidfVectorizer
#import pandas as pd
from scipy.sparse import coo_matrix
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

#Umformatierung von dictionary in pd dataframe
def convert(data_from_dict, split):
    list = []
    for i in range(len(train)):
        list.append(data_from_dict[i][split])
    return list

train_text = convert(train, 'text')
train_label = convert(train, 'label')
test_text = convert(test, 'text')
test_label = convert(test, 'label')


# Vec transform the text with count vectorizer
vec = CountVectorizer(ngram_range=(1,1), lowercase=True)
trn_x = vec.fit_transform(train_text)
tst_x = vec.transform(test_text)


# Convert csr matrices to sparse format
trn_x_coo = coo_matrix(trn_x)
tst_x_coo = coo_matrix(tst_x)

trn_values = trn_x_coo.data
trn_indices = np.vstack((trn_x_coo.row, trn_x_coo.col))
tst_values = tst_x_coo.data
tst_indices = np.vstack((tst_x_coo.row, tst_x_coo.col))

trn_i = torch.LongTensor(trn_indices)
trn_v = torch.FloatTensor(trn_values)
trn_shape = trn_x_coo.shape
tst_i = torch.LongTensor(tst_indices)
tst_v = torch.FloatTensor(tst_values)
tst_shape = tst_x_coo.shape

#Making the test and train tensors for the text
trn_x_tensor = torch.sparse.FloatTensor(trn_i, trn_v, torch.Size(trn_shape))
tst_x_tensor = torch.sparse.FloatTensor(tst_i, tst_v, torch.Size(tst_shape))

#Making y which is the tensor of emotion labels
y = torch.tensor(test_label)

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
opt = torch.optim.Adam(model.parameters(), lr=1e-5)


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
epochs = 100 # if choosing smaller batches go for less epochs
fit(epochs, model, loss_fn, opt, train_dl)
torch.save(model, "model.pth")
#model = torch.load("model700e_1e-4wd.pth")


# TEST THE MODEL
pred_test = model(tst_x_tensor)

# Print results
p  = precision(pred_test, y_test, num_classes=6)
r  =    recall(pred_test, y_test, num_classes=6)
f1 =  f1_score(pred_test, y_test, num_classes=6)
print("F1-score:", f1)
print("Precision:", p)
print("Recall:", r)