import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# Device and data
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(torch.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(torch.relu(self.conv3(x)))  # 8x8 -> 4x4
        x = x.view(-1, 128 * 4 * 4)
        # x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Simple checkpoint save/load
def save_model(model, optimizer, epoch, losses, accuracies):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'losses': losses,
        'accuracies': accuracies
    }, 'checkpoint.pth')

def load_model(model, optimizer):
    if os.path.exists('checkpoint.pth'):
        checkpoint = torch.load('checkpoint.pth', map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Resumed from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'], checkpoint['losses'], checkpoint['accuracies']
    else:
        print("Starting fresh")
        return 0, [], []

# Data loading
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Model setup
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Load previous training if exists
start_epoch, losses, accuracies = load_model(model, optimizer)

# Training
print("Training...")
num_epochs = 30

for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Calculate accuracy
    train_acc = 100 * correct / total
    avg_loss = running_loss / len(trainloader)
    
    # Test accuracy every few epochs
    if epoch % 5 == 0:
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_acc = 100 * test_correct / test_total
        print(f"Epoch {epoch+1}: Loss={avg_loss:.3f}, Train Acc={train_acc:.1f}%, Test Acc={test_acc:.1f}%")
    else:
        print(f"Epoch {epoch+1}: Loss={avg_loss:.3f}, Train Acc={train_acc:.1f}%")
    
    # Store metrics
    losses.append(avg_loss)
    accuracies.append(train_acc)
    
    # Save every 5 epochs
    if epoch % 5 == 0:
        save_model(model, optimizer, epoch, losses, accuracies)

# Final test
print("\nFinal Testing...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"Final Test Accuracy: {test_acc:.1f}%")

# Save final model
save_model(model, optimizer, num_epochs-1, losses, accuracies)

# Simple plot
if losses:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    plt.show()

# Simple Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Get predictions (reuse the data from final test)
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu())
        all_labels.extend(labels.cpu())

# Plot confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap='Blues')
plt.title(f'Confusion Matrix - {test_acc:.1f}% Accuracy')
plt.tight_layout()
plt.show()

print("Done! Model saved to checkpoint.pth")