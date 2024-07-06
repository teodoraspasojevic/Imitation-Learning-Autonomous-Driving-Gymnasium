from data import CarDataset, ResizeWithLabels, RandomHorizontalFlipWithLabel, RandomVerticalFlipWithLabel, RandomRotationWithLabel, RGBTo3CGrayscale, ToTensorWithLabel, ComposeTransformations
from model import Model, RecurrentModel, to_device, train_model, test_model, plot_history
from torch.utils.data import DataLoader, Subset
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim


def compute_class_weights(train_dataset):
    class_counts = Counter()
    for _, labels in train_dataset:
        for index, label in enumerate(labels):
            if label == 1:
                class_counts[index] += 1

    # Compute weights inversely proportional to class frequencies.
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    # Normalize weights.
    total_weight = sum(class_weights.values())
    class_weights = {cls: weight / total_weight for cls, weight in class_weights.items()}

    # Convert weights to tensor.
    weights = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float32)

    return weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = ComposeTransformations([
    ResizeWithLabels(),
    RandomHorizontalFlipWithLabel(prob=0.2),
    RandomVerticalFlipWithLabel(prob=0.2),
    RandomRotationWithLabel(degrees=20, prob=0.2),
    # RGBTo3CGrayscale(),
    ToTensorWithLabel()
])

car_dataset = CarDataset(root='../../data_more_commands', transform=transforms)

train_size = int(0.6 * len(car_dataset))
val_size = int(0.2 * len(car_dataset))
test_size = len(car_dataset) - train_size - val_size

train_indices = list(range(train_size))
val_indices = list(range(train_size, train_size + val_size))
test_indices = list(range(train_size + val_size, len(car_dataset)))

train_dataset = Subset(car_dataset, train_indices)
val_dataset = Subset(car_dataset, val_indices)
test_dataset = Subset(car_dataset, test_indices)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

model = RecurrentModel()

# weights = compute_class_weights(train_dataset)
criterion = nn.CrossEntropyLoss()
# weights = weights.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model, history, best_epoch = train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, epochs=20)

plot_history(history, best_epoch)
