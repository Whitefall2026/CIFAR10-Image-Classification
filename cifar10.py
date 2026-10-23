import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.utils.data import ConcatDataset



#==========HyperParameters=========

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 80

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device: ", device)

#==========LoadingDataseets=========

mean = (0.4914, 0.4822, 0.4465)

std = (0.2023, 0.1944, 0.2010)




train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_dataset = torchvision.datasets.CIFAR10(
    root = "./data",
    train = True,
    download = True,
    transform = train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root = "./data",
    train = False,
    download = True,
    transform = test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


classes = (
    "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
)

print("train_dataset: ", len(train_dataset))
print("test_dataset: ", len(test_dataset))


#==========DefiningCNN=========


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 10)

        )

    def forward(self, x):
        x = self.features(x)

        x = self.classifier(x)
        return x


model = NeuralNetwork().to(device)


#==========Loss&Optimizer=========

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr = LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


#==========Training=========
max_test_accuracy = 0.0


for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    train_accuracy = 100 * correct / total


    print(
        f"\nEpoch[{epoch + 1}/{EPOCHS}]"
        f"\nLoss: {epoch_loss:.4f}"
        f"Train Accuracy: {train_accuracy:.4f}%"
    )


    #==========Test=========


    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_accuracy = 100 * correct / total

    if test_accuracy > max_test_accuracy:
        max_test_accuracy = test_accuracy
        best_illiterations = epoch

    print(f"TestAccuracy: {test_accuracy:.4f}%")
print(f"Max Test Accuracy: {max_test_accuracy:.4f}% in Epoch {best_illiterations + 1}")

#==========SaveParameters=========

torch.save(model.state_dict(), "cifar10_cnn.pth")
print("Model saved to cifar10_cnn.pth")





