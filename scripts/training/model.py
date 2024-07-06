import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss does not improve.

    Attributes:
        patience (int): Number of epochs to wait after the last time validation loss improved.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        counter (int): Counter for epochs with no improvement.
        best_loss (float): Best validation loss observed.
        early_stop (bool): Flag to indicate if early stopping should be performed.
    """

    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Checks if early stopping should be performed based on the validation loss.

        Args:
            val_loss (float): Current validation loss.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class Model(nn.Module):
    """
   Convolutional Neural Network model for image classification.

   Attributes:
       conv1 (nn.Conv2d): First convolutional layer.
       pool1 (nn.MaxPool2d): First max pooling layer.
       conv2 (nn.Conv2d): Second convolutional layer.
       pool2 (nn.MaxPool2d): Second max pooling layer.
       fc1 (nn.Linear): First fully connected layer.
       fc2 (nn.Linear): Second fully connected layer.
   """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

class RecurrentModel(nn.Module):
    """
   Convolutional Neural Network model for image classification.

   Attributes:
       conv1 (nn.Conv2d): First convolutional layer.
       pool1 (nn.MaxPool2d): First max pooling layer.
       conv2 (nn.Conv2d): Second convolutional layer.
       pool2 (nn.MaxPool2d): Second max pooling layer.
       lstm (nn.LSTM): LSTM for learning temporal dependencies.
       fc (nn.Linear): Fully connected layer as a classification head.
   """

    def __init__(self, hidden_dim=128, num_layers=1, sequence_length=16):
        super().__init__()
        self.sequence_length = sequence_length
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(input_size=64 * 24 * 24, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(128, 5)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # Reshape the tensor to (batch_size, sequence_length, input_size)
        original_batch_size = x.size(0)
        sequence_length = self.sequence_length
        batch_size = original_batch_size // sequence_length
        x = x[:batch_size * sequence_length]
        x = x.view(batch_size, sequence_length, 64 * 24 * 24)

        if batch_size == 0:
            x = torch.zeros(original_batch_size, 128)
            return x

        x, _ = self.lstm(x)

        x = x.contiguous().view(batch_size * sequence_length, -1)

        x = self.fc(x)
        if original_batch_size % sequence_length != 0:
            padding_size = original_batch_size % sequence_length
            padding = torch.zeros(padding_size, x.size(1))
            x = torch.cat((x, padding), dim=0)

        return x


def to_device(data, device):
    """
    Moves a tensor or a collection of tensors to the specified device.

    Args:
        data (torch.Tensor or list/tuple): Input tensor or collection of tensors.
        device (torch.device): Device to move the data to.

    Returns:
        torch.Tensor or list/tuple: Data moved to the specified device.
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, epochs=10):
    """
    Trains the model and evaluates on validation and test datasets.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        epochs (int): Number of epochs to train the model. Defaults to 10.

    Returns:
        model (nn.Module): The trained model.
        history (dict): Dictionary containing training history.
        best_epoch (int): The epoch with the best validation accuracy.
    """

    # Save the training history.
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'test_accuracy': []
    }
    # Move the model to GPU.
    model.to(device)

    best_val_accuracy = 0
    best_epoch = 0

    # Use early stopping.
    early_stopping = EarlyStopping(patience=3, min_delta=0.0)

    for epoch in range(epochs):

        # Train the model.
        model.train()

        train_total_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Ensure labels are in the correct shape for CrossEntropyLoss
            if labels.ndimension() == 2:
                labels = torch.argmax(labels, dim=1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels_class = torch.max(labels, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels_class).sum().item()

        train_loss = train_total_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # Evaluate the model.
        model.eval()

        val_total_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, labels_class = torch.max(labels, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels_class).sum().item()

            val_loss = val_total_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total

            # Save the best epoch.
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                torch.save(model.state_dict(), '../../models/best_model.pth')

        # Test the model.
        test_accuracy = test_model(model, test_loader)

        # Save training history.
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['test_accuracy'].append(test_accuracy)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

        early_stopping(val_loss)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model, history, best_epoch


def test_model(model, test_loader):
    """
    Test the model on the test dataset and return accuracy.

    Args:
    model (torch.nn.Module): The model to be tested.
    test_loader (torch.utils.data.DataLoader): The test data loader.

    Returns:
    float: The test accuracy.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, labels_class = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == labels_class).sum().item()

    test_accuracy = 100 * correct / total
    return test_accuracy


def load_best_model(model, checkpoint_path):
    """
    Load the best model from the checkpoint.

    Args:
    model (torch.nn.Module): The model to load the weights into.
    checkpoint_path (str): Path to the checkpoint file.

    Returns:
    model (torch.nn.Module): The model with loaded weights.
    """
    model.load_state_dict(torch.load(checkpoint_path))
    return model


def plot_history(history, best_epoch):
    """
    Plot training, validation, and test accuracies and losses.

    Args:
        history (dict): Dictionary containing training history.
        best_epoch (int): The epoch with the best validation accuracy.
    """
    train_losses = history['train_loss']
    val_losses = history['val_loss']
    train_accuracies = history['train_accuracy']
    val_accuracies = history['val_accuracy']
    test_accuracies = history['test_accuracy']

    epochs = range(1, len(train_losses) + 1)

    # Plotting training and validation losses.
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting training, validation and test accuracies.
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.plot(epochs, test_accuracies, 'g-', label='Test Accuracy')
    plt.title('Training, Validation and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.show()

    print(f'Best epoch was: {best_epoch}')


# def get_confusion_matrix(data_loader, model):
#     """
#     Computes the confusion matrix for a given model and data loader.
#
#     Args:
#        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate.
#        model (torch.nn.Module): The model to evaluate.
#
#     Returns:
#        numpy.ndarray: Confusion matrix of shape (num_classes, num_classes).
#     """
#     model.eval()
#     all_targets = []
#     all_predictions = []
#     with torch.no_grad():
#         for images, targets in data_loader:
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             all_targets.extend(targets)
#             all_predictions.extend(predicted)
#
#     # Create the confusion matrix
#     cm = confusion_matrix(all_targets, all_predictions)
#
#     return cm
