from torch import nn

class Classifier(nn.Module):
    def __init__(self, hidden_layer, input_layer):
        super(Classifier, self).__init__()
        
        self.hidden = nn.Linear(input_layer, hidden_layer)
        self.output = nn.Linear(hidden_layer, 102)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.logsoftmax(x)
        
        return x