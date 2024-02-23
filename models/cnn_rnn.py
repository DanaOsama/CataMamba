import torch
import torch.nn as nn
import torchvision.models as models

CNN_models = {"resnet18": 512, "resnet50": 2048, "resnet101": 2048}


class CNN_RNN_Model(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers, cnn="resnet50", rnn="lstm", bidirectional=False):
        super(CNN_RNN_Model, self).__init__()

        # Load a pre-trained CNN model
        if cnn == "resnet18":
            self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif cnn == "resnet50":
            self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif cnn == "resnet101":
            self.cnn = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else:
            raise ValueError("Invalid CNN model name")

        
        # Replace the classifier of the pre-trained model
        self.cnn.fc = nn.Identity()  # Use the CNN as a fixed feature extractor
        
        # Define the RNN layer
        if rnn == "lstm":
            self.rnn = nn.LSTM(input_size=CNN_models[cnn], hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True,
                           bidirectional=bidirectional)
        elif rnn == "gru":
            self.rnn = nn.GRU(input_size=CNN_models[cnn], hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          bidirectional=bidirectional)
        
        # Adjust the final fully connected layer to output predictions for each time step
        # If the RNN is bidirectional, it concatenates the hidden states from both directions,
        # so we multiply the hidden size by 2 for the input size to the fully connected layer.
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(rnn_output_size, num_classes)

        
    def forward(self, x):

        batch_size, sequence_length, C, H, W = x.size()

        x = x.view(batch_size * sequence_length, C, H, W)
        cnn_out = self.cnn(x)
        
        # Reshape the output back to (batch_size, sequence_length, cnn_output_size)
        cnn_out = cnn_out.view(batch_size, sequence_length, -1)
        
        # Pass the CNN's output to the RNN
        rnn_out, _ = self.rnn(cnn_out)
        
        # Instead of taking the output of the last time step,
        # we now process all time steps through the fully connected layer to get a prediction for each frame
        # rnn_out is of shape (batch_size, sequence_length, rnn_output_size)
        out = self.fc(rnn_out)
        
        # out is of shape (batch_size, sequence_length, num_classes)
        # This gives a prediction for each frame in the sequence

        return out