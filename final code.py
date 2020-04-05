#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing all the required libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.utils import np_utils


# #Loading the Data

# In[2]:


text=(open(r"C:\Users\USER\Desktop\sonnet.docx", errors='ignore').read())
text=text.lower()


# In[3]:


characters = sorted(list(set(text)))
n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}


# In[4]:


X = []
Y = []
length = len(text)
seq_length = 100
for i in range(0, length-seq_length, 1):
     sequence = text[i:i + seq_length]
     label =text[i + seq_length]
     X.append([char_to_n[char] for char in sequence])
     Y.append(char_to_n[label])


# In[5]:


X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)


# #Modelling

# In[6]:


model = Sequential()
model.add(LSTM(400, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# #Generating Text

# In[7]:


string_mapped = X[99]
# generating characters
for i in range(seq_length):
    x = np.reshape(string_mapped,(1,len(string_mapped), 1))
    x = x / float(len(characters))
    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_char[value] for value in string_mapped]
    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]


# In[ ]:




