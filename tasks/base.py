import torch

def _clip_grads(model):
    """Gradient clipping to the range [10, 10]."""
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.clamp_(-10, 10)

def _report(loss,cost):
    print("Loss: {}".format(loss))
    print("Cost: {}".format(cost))

class TaskBase(object):

    def __init__(self,name):
        self.name = name
        # self.loss = list()
        # self.cost = list()
    @property
    def model(self):
        raise NotImplementedError

    @property
    def data_gen(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
