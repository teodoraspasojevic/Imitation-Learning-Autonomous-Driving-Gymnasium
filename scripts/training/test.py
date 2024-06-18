from data import CarDataset, ResizeWithLabels, RandomHorizontalFlipWithLabel, RandomVerticalFlipWithLabel, ChangeStreetColor, ToTensorWithLabel, ComposeTransformations
from model import Model, test_model, load_best_model
from torch.utils.data import DataLoader, Subset

model = Model()
model = load_best_model(model, './best_model3.pth')

# Test the model on up-down flipped dataset.

transforms1 = ComposeTransformations([
    ResizeWithLabels(),
    RandomHorizontalFlipWithLabel(prob=0.5),
    ToTensorWithLabel()
])

car_dataset = CarDataset(root='../../data2', transform=transforms1)

train_size = int(0.6 * len(car_dataset))
val_size = int(0.2 * len(car_dataset))
test_size = len(car_dataset) - train_size - val_size

test_indices = list(range(train_size + val_size, len(car_dataset)))
test_dataset = Subset(car_dataset, test_indices)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

test_accuracy = test_model(model, test_loader)

print('###############################################')
print(f'Test accuracy on the first dataset: {test_accuracy:.4f}%')

# Test the model on left-right flipped dataset.

transforms2 = ComposeTransformations([
    ResizeWithLabels(),
    RandomVerticalFlipWithLabel(prob=0.5),
    ToTensorWithLabel()
])

car_dataset = CarDataset(root='../../data2', transform=transforms1)

train_size = int(0.6 * len(car_dataset))
val_size = int(0.2 * len(car_dataset))
test_size = len(car_dataset) - train_size - val_size

test_indices = list(range(train_size + val_size, len(car_dataset)))
test_dataset = Subset(car_dataset, test_indices)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

test_accuracy = test_model(model, test_loader)

print(f'Test accuracy on the second dataset: {test_accuracy:.4f}%')


# Test the model on brown road colour dataset.

transforms1 = ComposeTransformations([
    ResizeWithLabels(),
    ChangeStreetColor(),
    ToTensorWithLabel()
])

car_dataset = CarDataset(root='../../data2', transform=transforms1)

train_size = int(0.6 * len(car_dataset))
val_size = int(0.2 * len(car_dataset))
test_size = len(car_dataset) - train_size - val_size

test_indices = list(range(train_size + val_size, len(car_dataset)))
test_dataset = Subset(car_dataset, test_indices)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

test_accuracy = test_model(model, test_loader)

print(f'Test accuracy on the third dataset: {test_accuracy:.4f}%')

