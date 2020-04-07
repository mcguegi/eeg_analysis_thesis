# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 18:38:38 2020

@author: Camila
"""


import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

url = 'C:/Users/Camila/Documents/Tesis/csv/relative/data.csv'
eeg_dataset = pd.read_csv(url,error_bad_lines=False)

X = eeg_dataset[['alpha','betha','delta','gamma','theta']].values
y = eeg_dataset[['class']].values


x = torch.from_numpy(X).float().requires_grad_(True)

y = torch.from_numpy(y).float()

x.shape, y.shape

model = nn.Sequential(
          nn.Linear(5,1),
          nn.Sigmoid()
      )
loss_function = nn.BCELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

losses = []
iterations = 2000

for i in range(iterations):
  result = model(x)
  
  loss = loss_function(result, y)
  losses.append(loss.data)
  
  optimizer.zero_grad()
  loss.backward()
  
  optimizer.step()

plt.plot(range(iterations), losses)
loss


paciente_test = torch.Tensor([[0.08784774309110274,0.07233267054347474,0.658569550747688,0.04957907325400277,0.12731236330060244]])
prediction = "Epiléptico" if model(paciente_test) > 0.5 else "No Epiléptico"
print(model(paciente_test))

