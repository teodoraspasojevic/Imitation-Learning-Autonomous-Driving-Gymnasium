from data import CarDataset, RandomHorizontalFlipWithLabel, RandomVerticalFlipWithLabel, ChangeStreetColor, ToTensorWithLabel, ComposeTransformations
from model import Model, to_device, train_model, test_model, plot_history
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

transforms = ComposeTransformations([
    ToTensorWithLabel()
])

car_dataset = CarDataset(root='../../data', transform=transforms)

train_size = int(0.6 * len(car_dataset))
val_train_size = len(car_dataset) - train_size

train_dataset, val_test_dataset = random_split(dataset=car_dataset, lengths=[train_size, val_train_size])

val_size = int(0.5 * len(val_test_dataset))
test_size = len(val_test_dataset) - val_size

val_dataset, test_dataset = random_split(dataset=val_test_dataset, lengths=[val_size, test_size])

# We do not shuffle any dataset because we need frames to be consecutive.

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

model = Model()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)
