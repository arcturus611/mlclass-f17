#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:07:35 2017

@author: arcturus
"""

from __future__ import print_function
import torch
from torch.autograd import Variable
import numpy as np


x = torch.Tensor(5, 3)
print(x)
print(x.size())

y = torch.rand(5, 3)
print(y)

y = torch.rand(5, 3)
print(x + y) #1

print(torch.add(x, y)) #2

result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result) #3

# adds x to y
y.add_(x) #in place addn
print(y) #4

### Converting from torch to numpy is a breeze
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a = np.ones(5) #generate numpy array
b = torch.from_numpy(a) #torch the array
np.add(a, 1, out=a) # modify the numpy array
print(a) 
print(b) #see the torched array change too!! 


#### Autograd tutorial
x = Variable(torch.ones(2, 2), requires_grad=True) #create variable
print(x)

y = x + 2
print(y)

z = y*y*3

out = z.mean()

out.backward()

print(x.grad)

#######

x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
count = 0
while y.data.norm() < 1000:
    count+=1
    y = y * 2

print(y)




