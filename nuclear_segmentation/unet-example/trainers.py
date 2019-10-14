
# coding: utf-8

# In[1]:


import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np


# In[2]:


class Trainer:
    
    def __init__(self, model):
        self.model = model

    def fit_generator(self, dataset, criterion, optimizer, n_epochs=1, batch_size=1, shuffle=False):
        loss_history = []
                
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        for epoch in range(n_epochs):
            running_loss = 0.0
            
            print('Epoch : {}/{}'.format(epoch + 1, n_epochs))
            print('-'*10)
            
            for batch, (data, target) in enumerate(loader):
                data, target = Variable(data.cuda()), Variable(target.cuda())
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                running_loss += loss.item()
                
                if (batch % 10) == 9:
                    print('\tBatch : {}/{}\tLoss : {:.4f}'.format(batch+1, len(loader), loss.item()))
                
                loss.backward()
                optimizer.step()
            
            loss_history.append(running_loss/len(loader))
        return loss_history

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = Variable(x.unsqueeze(0).cuda())
        output = F.sigmoid(self.model(x)).data.cpu()
        return output.numpy()

    def predict_generator(self, dataset, batch_size=1):
        predictions = []
        loader = DataLoader(dataset, batch_size=batch_size)
        for batch, (data, target) in enumerate(loader):
            data = Variable(data.cuda())
            outputs = self.model(data)
            for prediction in outputs:
                predictions.append(F.sigmoid(prediction).data.cpu().numpy())
        return np.array(predictions)

