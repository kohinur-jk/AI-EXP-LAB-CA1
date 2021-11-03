# AI-EXP-LAB-CA1
House price estimation using baseline models, gradient boosting, and deep learning models
Here I have tried to implement deep learning model first, but the simple models output was same, I could not figure out what was the problem.
I want to be this kind of cool who answers the following way...
>>> import torch
>>> from torch import nn
>>> 
>>> class LinearRegressionModel(nn.Module):
...     def __init__(self, init_weights=False):
...         super(LinearRegressionModel, self).__init__()
...         self.hidden = torch.nn.Linear(19, 10, bias=True)
...         self.relu = torch.nn.ReLU()
...         self.predict = torch.nn.Linear(10, 1, bias=True)
...         if init_weights:
...             self._initialize_weights()
...     def forward(self, x):
...         x = self.hidden(x)
...         x = self.relu(x)
...         x = self.predict(x)
...         return x
... 
>>> 
>>> inp = torch.rand(5, 19)
>>> 
>>> model = LinearRegressionModel()
>>> out = model(inp)
>>> 
>>> out
tensor([[-0.2017],
        [-0.1873],
        [-0.0734],
        [-0.2187],
        [-0.1927]], grad_fn=<AddmmBackward>)
>>>


I did only try to implement what other people did, simple regression models.
I wanted to learn how to implement model ensemble but I did not learn it.
