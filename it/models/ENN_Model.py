import torch
import torch.nn as nn
import torch.optim as optim

class Model_Ethics(nn.Module):
    # Encoder 40->40  40->400
    #--Business Expert 400->2
    #--Ethics Expert 400->50
    #-Concat(50|2)
    #Ethics Aware 52->50
    def __init__(self, input_sizes, output_sizes, dropout_rate):
        super().__init__()

        #encoder
        self.layer_input = nn.Linear(in_features = input_sizes[0], out_features = output_sizes[0],  bias=True )
        self.layer_output = nn.Linear(in_features = input_sizes[1], out_features = output_sizes[1],  bias=True )
        nn.init.constant_(self.layer_input.bias, 0)
        nn.init.constant_(self.layer_output.bias, 0)
        # nn.init.zeros_(self.layer_input.weight)
        # nn.init.zeros_(self.layer_output.weight)
        torch.nn.init.xavier_uniform_(self.layer_input.weight)
        torch.nn.init.xavier_uniform_(self.layer_output.weight)
        

        #business
        self.layer_business = nn.Linear(in_features = input_sizes[2], out_features = output_sizes[2],  bias=True )
        nn.init.constant_(self.layer_business.bias, 0)
        # nn.init.zeros_(self.layer_business.weight)
        torch.nn.init.xavier_uniform_(self.layer_business.weight)

        #Ethics Expert
        self.layer_ethics_E = nn.Linear(in_features = input_sizes[3], out_features = output_sizes[3],  bias=True )
        nn.init.constant_(self.layer_ethics_E.bias, 0)
        # nn.init.zeros_(self.layer_ethics_E.weight)
        torch.nn.init.xavier_uniform_(self.layer_ethics_E.weight)

        #Ethics Aware
        self.layer_ethics_A = nn.Linear(in_features = input_sizes[4], out_features = output_sizes[4],  bias=True )
        nn.init.constant_(self.layer_ethics_A.bias, 0)
        # nn.init.zeros_(self.layer_ethics_A.weight)
        torch.nn.init.xavier_uniform_(self.layer_ethics_A.weight)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        #self.softmax = nn.Softmax()

    def forward(self, x):

        #Encoder
        x = self.dropout(self.relu(self.layer_input(x)))
        x0 = self.dropout(self.relu(self.layer_output(x)))

        #BE
        x1 = self.layer_business(x0)

        #EE
        x2 = self.relu(self.layer_ethics_E(x0))

        #Concat
        x_concat = torch.cat([x1, x2], 1)
        
        #EA
        x3 = self.dropout(self.layer_ethics_A(x_concat))

        return x1, x2, x3


def init_ENN_model(dataset_name, number_of_features, LEARNING_RATE):

    nof = number_of_features

    # if dataset_name == "german_credit":
    #     model = Model_Ethics(input_sizes = [40, 400, 400, 400, 52], output_sizes = [400, 400, 2, 50, 50], dropout_rate=0.2)
    # else:
    model = Model_Ethics(input_sizes = [nof, nof, nof*10, nof*10, 52], output_sizes = [nof, nof*10, 2, 50, 50], dropout_rate=0.2)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterionBE = nn.CrossEntropyLoss()
    criterionEE_EA = nn.MSELoss()

    return model, optimizer, criterionBE, criterionEE_EA



