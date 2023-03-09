from torch import nn, argmax, tensor, Tensor

class AverageAccuracy(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.total_data = 0
        self.total_correct = 0
        self.softmax = nn.Softmax(1)

    def forward(self, x, y: Tensor):
        if(self.size == self.total_data):
            self.total_correct = 0
            self.total_data = 0

        if(len(y.shape) > 1 and y.shape[-1] == 1):
            y = y.reshape(y.shape[:-1])
            
        self.total_data += x.shape[0]
        x = self.softmax(x)
        x = argmax(x, 1)
        correct = (x == y).sum().item()
        self.total_correct += correct

        return tensor([100*self.total_correct/self.total_data])

