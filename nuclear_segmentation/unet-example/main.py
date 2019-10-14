
# coding: utf-8

# In[1]:


import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt

from models import Unet
from datasets import ISBI2012Dataset
from trainers import Trainer

import warnings
warnings.filterwarnings("ignore")


# In[2]:


transform = transforms.Compose([transforms.ToTensor()])


# In[3]:


isbi = ISBI2012Dataset('./dataset/train-volume.tif', './dataset/train-labels.tif',
                       transforms=transform)


# In[4]:


isbi.train.shape


# In[5]:


isbi.targets.shape


# In[6]:


plt.imshow(isbi.train[0].reshape(512, 512), cmap='gray');


# In[7]:


plt.imshow(isbi.targets[0].reshape(512, 512) , cmap='gray');


# In[8]:


unet = Unet()
unet.cuda();


# In[9]:


trainer = Trainer(unet)


# In[10]:


criterion = nn.BCEWithLogitsLoss()


# In[11]:


optimizer = optim.Adam(trainer.model.parameters(), lr=1e-3)
loss_history = trainer.fit_generator(isbi, criterion, optimizer, 25);


# In[12]:


loss_history


# In[13]:


img, target = isbi[0]
y_pred = trainer.predict(img)


# In[14]:


y_pred


# In[15]:


plt.imshow(y_pred.reshape(512, 512), cmap='gray');


# In[16]:


thresh = 0.5
y_pred[y_pred >= thresh] = 1
y_pred[y_pred < thresh] = 0
y_pred


# In[17]:


plt.imshow(y_pred.reshape(512, 512), cmap='gray');


# In[18]:


plt.imshow(target.numpy().reshape(512, 512), cmap='gray');


# In[19]:


preds = trainer.predict_generator(isbi)


# In[20]:


preds.shape


# In[21]:


preds[0]


# In[22]:


for y_pred in preds:
    plt.figure()
    plt.imshow(y_pred.reshape(512, 512), cmap='gray')


# In[23]:


thresh = 0.5
for y_pred in preds:
    y_pred[y_pred >= thresh] = 1
    y_pred[y_pred < thresh] = 0


# In[24]:


preds[0]


# In[25]:


for y_pred in preds:
    plt.figure()
    plt.imshow(y_pred.reshape(512, 512), cmap='gray')

