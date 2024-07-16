from data import CarDataset, ResizeWithLabels, ToTensorWithLabel, ComposeTransformations
from model import Model, load_best_model, quantize_model
from torch.utils.data import DataLoader, Subset

import time
import torch
import os
from ptflops import get_model_complexity_info


def measure_inference_time(model, input_data):
    """
    Measures the inference time of the model on the given input data.

    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        input_data (torch.Tensor): Input tensor for which inference time is measured.

    Returns:
        float: The time taken for the model to perform inference in seconds.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        _ = model(input_data.to(device))
        end_time = time.time()
    return end_time - start_time


def measure_flops(model, input_res=(3, 96, 96)):
    """
    Measures the number of floating point operations (FLOPs) required by the model.

    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        input_res (tuple): The shape of the input tensor (default is (3, 96, 96)).

    Returns:
        tuple: The number of FLOPs and parameters as strings.
    """
    flops, params = get_model_complexity_info(model, input_res, as_strings=True, print_per_layer_stat=False)
    return flops, params


def measure_model_size(model_path):
    """
    Measures the size of the model stored at the given path.

    Args:
        model_path (str): Path to the model file.

    Returns:
        float: The size of the model file in megabytes (MB).
    """
    size_in_mb = os.path.getsize(model_path) / (1024 * 1024)
    return size_in_mb


def measure_accuracy(model, dataloader):
    """
    Measures the accuracy of the model on the given DataLoader.

    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the evaluation data.

    Returns:
        float: The accuracy of the model as a fraction of correct predictions.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, labels_class = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == labels_class).sum().item()
    return correct / total


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = ComposeTransformations([
        ResizeWithLabels(),
        ToTensorWithLabel()
    ])

    # Load the test dataset.
    car_dataset = CarDataset(root='../../data', transform=transforms)

    train_size = int(0.6 * len(car_dataset))
    val_size = int(0.2 * len(car_dataset))
    test_size = len(car_dataset) - train_size - val_size

    test_indices = list(range(train_size + val_size, len(car_dataset)))
    test_dataset = Subset(car_dataset, test_indices)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

    # Load original model.
    model_path = '../../models/best_model_augment.pth'
    model = Model()
    model = load_best_model(model, model_path)

    # Measure original model.
    input_data, _ = next(iter(test_loader))
    original_inference_time = measure_inference_time(model, input_data)
    original_flops, _ = measure_flops(model)
    original_model_size = measure_model_size(model_path)
    original_accuracy = measure_accuracy(model, test_loader)

    print(f'Original Model - Inference Time: {original_inference_time:.6f} seconds')
    print(f'Original Model - FLOPS: {original_flops}')
    print(f'Original Model - Model Size: {original_model_size:.2f} MB')
    print(f'Original Model - Accuracy: {original_accuracy * 100:.2f}%')

    # Quantize the model.
    model = load_best_model(model, model_path)
    quantized_model = quantize_model(model)

    # Save quantized model to file to get its size.
    quantized_model_path = '../../models/quantized_model.pth'
    torch.save(quantized_model.state_dict(), quantized_model_path)

    # Measure quantized model.
    quantized_inference_time = measure_inference_time(quantized_model, input_data)
    quantized_flops, _ = measure_flops(quantized_model)
    quantized_model_size = measure_model_size(quantized_model_path)
    quantized_accuracy = measure_accuracy(quantized_model, test_loader)

    print(f'Quantized Model - Inference Time: {quantized_inference_time:.6f} seconds')
    print(f'Quantized Model - FLOPS: {quantized_flops}')
    print(f'Quantized Model - Model Size: {quantized_model_size:.2f} MB')
    print(f'Quantized Model - Accuracy: {quantized_accuracy * 100:.2f}%')
