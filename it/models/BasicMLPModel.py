import torch.nn as nn
import torch.optim as optim


class BasicMLPModel(nn.Module):
    def __init__(self, input_sizes, output_sizes, dropout_rate):
        super().__init__()
        
        #FOR EXAMPLE -> input_sizes = [40, 400, 400] and output_sizes = [400, 400, 2]

        self.layer_input = nn.Linear(in_features = input_sizes[0], out_features = output_sizes[0])
        self.layer_hidden = nn.Linear(in_features = input_sizes[1], out_features = output_sizes[1])
        self.layer_output = nn.Linear(in_features = input_sizes[2], out_features = output_sizes[2])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax()
        
    def forward(self, x):

        x = self.dropout(self.relu(self.layer_input(x)))
        x = self.dropout(self.relu(self.layer_hidden(x)))
        x = self.layer_output(x)
        #y_pred = self.softmax(x)

        return x


def init_MLP_model(number_of_features, LEARNING_RATE):

    nof = number_of_features

    #model = BasicMLPModel(input_sizes= [17, 170, 170], output_sizes=[170, 170, 2], dropout_rate=0.2)
    model = BasicMLPModel(input_sizes= [nof, nof*10, nof*10], output_sizes=[nof*10, nof*10, 2], dropout_rate=0.2)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion


