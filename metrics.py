from torch import nn, argmax

class AverageAccuracy(nn.Module):
    def __init__(self):
        self.total_data = 0
        self.total_correct = 0
        self.softmax = nn.Softmax(1)

    def forward(self, x, y):
        self.total_data += x.shape[0].item()
        x = self.softmax(x)
        x = argmax(x, 1)
        correct = (x == y).sum().item()
        self.total_correct += correct

        return 100*self.total_correct/self.total_data
