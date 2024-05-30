import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
import re
import ast


class ResizeWithLabels:
    """Resize the image to 96x96 dimension."""

    def __call__(self, img, label):
        """
        Args:
                img (PIL.Image): The image to be transformed.
                label: The label associated with the image.

        Returns:
            Tuple: Resized image and its (possibly reversed) label.
        """
        transform = transforms.Resize((96, 96))
        img = transform(img)
        return img, label


class RandomVerticalFlipWithLabel:
    """Randomly vertically flips an image with its label."""

    def __call__(self, img, label):
        """
        Args:
                img (PIL.Image): The image to be transformed.
                label: The label associated with the image.

        Returns:
            Tuple: The (possibly) vertically flipped image and its (possibly reversed) label.
        """
        if random.random() > 0.5:
            img = transforms.functional.vflip(img)
            # Convert command from to left to, to right.
            if label[2] == 1:
                label[2], label[3] = 0, 1
            # Convert command from to right to, to left.
            if label[3] == 1:
                label[2], label[3] = 1, 0
        return img, label


class RandomHorizontalFlipWithLabel:
    """Randomly horizontally flips an image with accordingly flipped label."""

    def __call__(self, img, label):
        """
        Args:
                img (PIL.Image): The image to be transformed.
                label: The label associated with the image.

        Returns:
            Tuple: The (possibly) horizontally flipped image and its label.
        """
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
        return img, label


class ChangeStreetColor:
    """Transforms grey areas in the image to brown.

    Attributes:
        brown_colour(tuple): RGB values of brown color to be set.
    """

    def __init__(self):
        self.brown_color = (139, 69, 19)

    def __call__(self, image, label):
        """
        Args:
                image (PIL.Image): The image to be transformed.
                label: The label associated with the image.

        Returns:
            Tuple: The transformed image and its label.
        """
        image_np = np.array(image)

        # Define the grey color range.
        lower_grey = np.array([100, 100, 100])
        upper_grey = np.array([200, 200, 200])

        # Create a mask for grey areas.
        mask = np.all(image_np >= lower_grey, axis=-1) & np.all(image_np <= upper_grey, axis=-1)

        # Change grey areas to brown.
        image_np[mask] = self.brown_color

        image = Image.fromarray(image_np)

        return image, label


class ToTensorWithLabel:
    """Converts an image to a tensor."""

    def __call__(self, image, label):
        """
        Args:
                img (PIL.Image): The image to be transformed.
                label: The label associated with the image.

        Returns:
        """
        image = transforms.ToTensor()(image)
        return image, label


class ComposeTransformations:
    """Applies the composed transformations to the image and its label."""

    def __init__(self, transformations):
        """
        Args:
                transformations (list): List of transformations to be composed.
        """
        self.transformations = transformations

    def __call__(self, image, label):
        """
        Args:
                img (PIL.Image): The image to be transformed.
                label: The label associated with the image.

        Returns:
            Tuple: The transformed image and its label.
        """
        for transform in self.transformations:
            image, label = transform(image, label)
        return image, label


class CarDataset(Dataset):
    """Custom dataset for loading car images and their labels.

    Attributes:
        root (str): Root directory of the dataset.
        images_path (str): Path to the directory containing images.
        label_path (str): Path to the CSV file containing labels.
        labels (pd.DataFrame): DataFrame containing image filenames and their corresponding labels.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.images_path = os.path.join(self.root, 'images')
        self.label_path = os.path.join(self.root, 'data_log.csv')
        self.labels = pd.read_csv(self.label_path, dtype={'image_filename': str, 'action': str})
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_path, self.labels.iloc[index]['image_filename'])
        image = Image.open(image_path).convert("RGB")
        label = self.labels.iloc[index]['action']
        label = np.array(re.findall(r"[-+]?\d*\.\d+|\d+", label), dtype=float)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    def visualize(self, num_samples=30):
        """Visualizes batch of images from the dataset with their labels.

        Args:
            num_samples(int): number of images to be visualized.

        """
        indices = np.random.choice(len(self), num_samples, replace=False)
        samples = [self[i] for i in indices]

        fig, axes = plt.subplots(5, 6, figsize=(15, 10))
        axes = axes.flatten()

        for ax, (img, label) in zip(axes, samples):
            ax.imshow(img)
            ax.set_title(label)
            ax.axis('off')

        plt.tight_layout()
        plt.show()


def safe_eval_list(row):
    try:
        return ast.literal_eval(row)
    except (ValueError, SyntaxError):
        # Handle the case where the list format is incorrect
        row = row.replace('.', ',')
        return ast.literal_eval(row)


def transform_row(row):
    # Initialize the new array with zeros
    new_array = [0, 0, 0, 0, 0]

    # Apply the given conditions
    if row[0] == -1:
        new_array[2] = 1
    if row[0] == 1:
        new_array[3] = 1
    if row[1] == 1:
        new_array[0] = 1
    if row[2] == 0.8:
        new_array[1] = 1
    if all(element == 0 for element in row):
        new_array[4] = 1

    return new_array


def modify_labels():
    file_path = '../../data/data_log.csv'
    df = pd.read_csv(file_path)

    # Process each row in the specified column
    transformed_actions = df['action'].apply(safe_eval_list).apply(transform_row)

    # Create a new DataFrame from the transformed data
    transformed_df = df[['timestamp', 'image_filename']].copy()
    transformed_df['action'] = transformed_actions.apply(str)

    # Save the new DataFrame to a CSV file
    transformed_df.to_csv('../../data/data_log_new.csv', index=False)


# Usage example.
if __name__ == '__main__':
    # Visualize original dataset.
    car_dataset = CarDataset(root='../../data', transform=None)
    car_dataset.visualize()

    # Visualize augmented dataset.
    augment1 = RandomHorizontalFlipWithLabel()
    augment2 = RandomVerticalFlipWithLabel()
    augment3 = ChangeStreetColor()
    augment4 = ResizeWithLabels()
    to_tensor = ToTensorWithLabel()

    augmentations = ComposeTransformations([
        augment1,
        augment2,
        augment3
        # to_tensor
    ])

    car_dataset_augmented = CarDataset(root='../../data', transform=augmentations)
    car_dataset_augmented.visualize()
