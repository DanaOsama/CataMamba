import torch
from tqdm import tqdm
from torchvision import models

MODELS = {
	"vgg16": models.vgg16(weights='VGG16_Weights.DEFAULT'),
	"vgg19": models.vgg19(weights='VGG19_Weights.DEFAULT'),
	"resnet": models.resnet50(weights='ResNet50_Weights.DEFAULT'),
	"mobilenet": models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT'),
}

def save_checkpoint(model, optimizer, epoch, path, best = False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    if(best):
        print("[INFO] best checkpoint saved at epoch {}".format(epoch))
    else:
        print("[INFO] checkpoint saved at epoch {}".format(epoch))
    
def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    print("[INFO] checkpoint loaded")
    return model

def train(model, optimizer, criterion, train_loader, DEVICE):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE) # move data to device
        optimizer.zero_grad()  # Zero the gradients
        labels = labels.float()
        predictions = model(inputs)

        loss = criterion(predictions, labels)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(train_loader)

# def validate(model, val_loader, DEVICE):
#     model.eval()  # Set the model to evaluation mode
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in tqdm(val_loader):
#             inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
#             labels = labels.float()

#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             print("predicted size:", predicted.size())
#             print("labels size:", labels.size())
#             correct += (predicted == labels).sum().item()
#     return correct/total * 100


def validate(model, validation_loader, DEVICE):
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_frames = 0

    with torch.no_grad():  # No need to track gradients during validation
        for data, labels in tqdm(validation_loader):
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass: compute the model output
            outputs = model(data)
            
            # Convert outputs to probabilities
            # Assuming outputs are raw scores (logits), use softmax for multi-class classification
            # TODO: Check the softmax thing
            # probabilities = torch.softmax(outputs, dim=-1)
            
            # Convert probabilities to predicted class indices
            _, predicted_indices = torch.max(outputs, dim=-1)
            
            # Convert one-hot encoded labels to class indices for comparison
            _, labels_indices = torch.max(labels, dim=-1)
            
            # Calculate accuracy
            correct = (predicted_indices == labels_indices).sum().item()
            total_correct += correct
            total_frames += labels.size(0) * labels.size(1)  # Multiply batch size by sequence length

    accuracy = total_correct / total_frames
    print(f'Validation Accuracy: {accuracy:.4f}')
    return accuracy

# Example usage:
# model = YourModel(...)
# validation_dataset = YourDataset(...)
# validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
# validate_model(model, validation_loader, device='cuda' or 'cpu')
