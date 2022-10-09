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
sample = ['''I know your eyes in the morning sun
I feel you touch me in the pouring rain
And the moment that you wander far from me
I wanna feel you in my arms again
And you come to me on a summer breeze
Keep me warm in your love, then you softly leave
And it's me you need to show
How deep is your love
How deep is your love, how deep is your love
I really mean to learn
'Cause we're living in a world of fools
Breaking us down when they all should let us be
We belong to you and me
I believe in you
You know the door to my very soul
You're the light in my deepest, darkest hour
You're my savior when I fall
And you may not think, I care for you
When you know down inside that I really do
And it's me you need to show
How deep is your love
How deep is your love, how deep is your love
I really mean to learn
'Cause we're living in a world of fools
Breaking us down when they all should let us be
We belong to you and me
And you come to me on a summer breeze
Keep me warm in your love, then you softly leave
And it's me you need to show
How deep is your love
How deep is your love, how deep is your love
I really mean to learn
'Cause we're living in a world of fools
Breaking us down when they all should let us be
We belong to you and me
How deep is your love, how deep is your love
I really mean to learn
'Cause we're living in a world of fools
Breaking us down when they all should let us be
We belong to you and me
How deep is your love, how deep is your love
I really mean to learn
'Cause we're living in a world of fools
Breaking us down when they all should let us be
We belong to you and me''']


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
    text_list = text_list[0].split('\n')
    return text_list

#sample = text_to_list_of_sentences(sample)


def pred_validation(pred):
    Sum_of_emotions = []
    for w in pred:
        sum = torch.sum(w)
        list = w.tolist()
        sum = sum.item()
        for x in range(len(list)):
            list[x] = round(list[x] / sum, 4)
        Sum_of_emotions.append(list)
    return Sum_of_emotions

def sum_over_parts(list_of_parts):
    sums_of_emotions = []
    for emotion in range(6):
        sum = 0
        for part in list_of_parts:
            sum = sum + part[emotion]
        sums_of_emotions.append(sum)
    overall_sum = 0
    for sum in sums_of_emotions:
        overall_sum = overall_sum + sum
    for i in range(6):
        sums_of_emotions[i] = round(sums_of_emotions[i] / overall_sum, 4)
            
    return sums_of_emotions

#Daten zu one-hot wordvector machen
vec = CountVectorizer(ngram_range=(1,1), lowercase=True)
trn_x = vec.fit_transform(train_text)

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
sample_x_tensor = text_to_list_of_sentences(sample)
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
        self.fc4 = nn.Linear(1024, 1024)
        self.relu4 = nn.ReLU()
        self.dout4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(1024, 64)
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
        #a4 = self.fc4(dout3)
        #h4 = self.relu4(a3)
        #dout4 = self.dout4(h3)
        a5 = self.fc5(dout1)
        h5 = self.prelu(a5)
        a6 = self.out(h5)
        y = self.out_act(a6)
        return y

#Initialize model      
model = Net()


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
#fit(epochs, model, loss_fn, opt, train_dl)
#torch.save(model, "model.pth")
model = torch.load("model700e_1e-4wd.pth")


# TEST THE MODEL
pred_test = model(tst_x_tensor)
pred_sample = model(sample_x_tensor)
pred_percentage = pred_validation(pred_sample)
#print(pred_percentage)
final_verdict = sum_over_parts(pred_percentage)
#print(final_verdict)
print('Sadness: {} || Joy: {} || Love: {} || Anger: {} || Fear: {} || Suprise: {}'.format(final_verdict[0], final_verdict[1], final_verdict[2], final_verdict[3], final_verdict[4], final_verdict[5]))
"""
# Print results
p  = precision(pred_test, y_test, num_classes=6)
r  =    recall(pred_test, y_test, num_classes=6)
f1 =  f1_score(pred_test, y_test, num_classes=6)
print("F1-score:", f1)
print("Precision:", p)
print("Recall:", r)
"""

