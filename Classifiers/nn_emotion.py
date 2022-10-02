

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
test = load_dataset('emotion', split='train')

#Umnformatierung von dictionary in pd dataframe
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

# Define Dataloader
batch_size = 1000 #batchsize depends on available memory
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(trn_x_tensor.shape[1] , 1024)
        self.relu1 = nn.ReLU()
        self.dout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(1024, 2048)
        self.relu2 = nn.ReLU()
        self.dout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(2048, 1024)
        self.relu3 = nn.ReLU()
        self.dout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(1024, 64)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(64, 6)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout1 = self.dout1(h1)
        a2 = self.fc2(dout1)
        h2 = self.relu2(a2)
        dout2 = self.dout2(h2)
        a3 = self.fc3(dout2)
        h3 = self.relu3(a3)
        dout3 = self.dout3(h3)
        a4 = self.fc4(dout1)
        h4 = self.prelu(a4)
        a5 = self.out(h4)
        y = self.out_act(a5)
        return y
        
model = Net()


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
            #setting right dtype for loss_fn (long required)
            yb = torch.tensor(yb, dtype=torch.long)

            # 1. Generate predictions
            pred = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 1 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

epochs = 100 #if choosing smaller batches go for less epochs
fit(epochs, model, loss_fn, opt, train_dl)
torch.save(model, "model.pth")
#model = torch.load("model.pth")

pred_test = model(tst_x_tensor)



p = precision(pred_test, y, num_classes=6)
r = recall(pred_test, y, num_classes=6)
f1 = f1_score(pred_test, y, num_classes=6)


print("F1-score:", f1)
print("Precision:", p)
print("Recall:", r)


