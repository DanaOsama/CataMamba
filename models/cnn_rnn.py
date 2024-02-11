import torch
import torch.nn as nn
import torchvision.models as models


# class CNN_RNN_Model(nn.Module):
#     def __init__(self, num_classes, hidden_size, num_layers, bidirectional=False):
#         super(CNN_RNN_Model, self).__init__()
        
#         # Load a pre-trained CNN model
#         self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
#         # Replace the classifier of the pre-trained model
#         # Assuming the feature size output by the CNN is 512
#         self.cnn.fc = nn.Identity()  # Use the CNN as a fixed feature extractor
        
#         # Define the RNN layer
#         self.rnn = nn.LSTM(input_size=512, hidden_size=hidden_size,
#                            num_layers=num_layers, batch_first=True,
#                            bidirectional=bidirectional)
        
#         # Define the final fully connected layer
#         self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)
        
#     def forward(self, x):
#         # x is of shape (batch_size, sequence_length, C, H, W)
#         batch_size, sequence_length, C, H, W = x.size()
        
#         # Flatten the first two dimensions to treat the entire batch as independent
#         x = x.view(batch_size * sequence_length, C, H, W)
        
#         # Pass the input through the CNN
#         cnn_out = self.cnn(x)
        
#         # Reshape the output to (batch_size, sequence_length, cnn_output_size)
#         cnn_out = cnn_out.view(batch_size, sequence_length, -1)
        
#         # Pass the CNN's output to the RNN
#         rnn_out, _ = self.rnn(cnn_out)
        
#         # Take the output of the last time step
#         rnn_out = rnn_out[:, -1, :]
        
#         # Pass the RNN's output through the final fully connected layer
#         out = self.fc(rnn_out)
        
#         return out


class CNN_RNN_Model(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers, bidirectional=False):
        super(CNN_RNN_Model, self).__init__()
        
        # Load a pre-trained CNN model
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Replace the classifier of the pre-trained model
        self.cnn.fc = nn.Identity()  # Use the CNN as a fixed feature extractor
        
        # Define the RNN layer
        self.rnn = nn.LSTM(input_size=512, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True,
                           bidirectional=bidirectional)
        
        # Adjust the final fully connected layer to output predictions for each time step
        # If the RNN is bidirectional, it concatenates the hidden states from both directions,
        # so we multiply the hidden size by 2 for the input size to the fully connected layer.
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(rnn_output_size, num_classes)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        # x is of shape (batch_size, sequence_length, C, H, W)
        batch_size, sequence_length, C, H, W = x.size()
        
        # Process each frame through the CNN
        # Flatten the first two dimensions to treat each frame as independent
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

        # TODO: Add a softmax layer here
        out = self.softmax(out)
        return out

