import os
import itertools
import typing
import PIL.Image
import torch
import torchvision
import PIL

class ImageFolder(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset to handle image data organized in a specific folder structure.

    Args:
        folder (str): Path to the folder containing the images.
        transforms (typing.Callable): A callable for applying transformations to the images.
    """
    def __init__(self, folder: str, transforms: typing.Callable) -> None:
        self._files = []  # Initialize an empty list for storing image paths and labels
        self._transforms = transforms  # Store the transformations

        # Iterate over all files in the folder to classify them into groups
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)  # Create the full file path
            
            # Extract the filename without extension
            filename_without_extension = os.path.splitext(filename)[0]

            # Skip files that end with "DAPI" or "TRANS" before the file extension
            if filename_without_extension.lower().endswith("dapi") or filename_without_extension.lower().endswith("trans"):
                continue

            # Classify images into groups based on filename prefixes
            if filename.lower().startswith("0min"):
                self._files.append((0, filepath))  # Group 0: 0 minutes
            elif filename.lower().startswith("5min"):
                self._files.append((1, filepath))  # Group 1: 5 minutes
            elif filename.lower().startswith("15min"):
                self._files.append((2, filepath))  # Group 2: 15 minutes
            elif filename.lower().startswith("30min"):
                self._files.append((3, filepath))  # Group 3: 30 minutes

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self._files)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'group' (int): The group label of the image.
                - 'image' (torch.Tensor): The transformed image tensor.
        """
        # Get the group label and file path for the specified index
        group, filepath = self._files[idx]

        # Open the image, convert it to RGB, and apply transformations
        image = self._transforms(PIL.Image.open(filepath).convert(mode='RGB'))

        # Return a dictionary containing the label and transformed image
        return {'group': group, 'image': image}


def main() -> None:
    """
    Main function to load the dataset, configure the model, and train using backpropagation.
    """
    # Determine the computation device (use GPU if available, otherwise use CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the data augmentation and preprocessing pipeline
    transforms_pipeline = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=256),  # Resize images to 256x256 pixels
            torchvision.transforms.RandomAffine(
                degrees=180,  # Allow random rotations up to 180 degrees
                scale=(0.9, 1.11)  # Random scaling within the given range
            ),
            torchvision.transforms.RandomHorizontalFlip(),  # Random horizontal flipping
            torchvision.transforms.RandomVerticalFlip(),  # Random vertical flipping
            torchvision.transforms.RandomCrop(size=256),  # Random cropping to 256x256
            torchvision.transforms.ToTensor(),  # Convert image to a PyTorch tensor
            torchvision.transforms.Normalize(
                mean=(0.5, 0.5, 0.5),  # Normalize channels to have mean 0.5
                std=(0.5, 0.5, 0.5)  # Normalize channels to have std 0.5
            )
        ]
    )

    # Create an instance of the custom ImageFolder dataset
    data_source = ImageFolder(
        folder="./data_unpack",  # Path to the folder containing images
        transforms=transforms_pipeline  # Data transformation pipeline
    )

    # Configure the DataLoader for batch processing
    data_loader = torch.utils.data.DataLoader(
        dataset=data_source,
        batch_size=16,  # Number of samples per batch
        shuffle=True,  # Shuffle the dataset at every epoch
        num_workers=4,  # Number of worker threads for data loading
        persistent_workers=True,  # Keep worker processes alive for reuse
        pin_memory=True  # Pin memory to speed up data transfer to GPU
    )

    # Load the ResNet-18 model with pretrained weights
    model = torchvision.models.resnet18(weights='DEFAULT')

    # Replace the final fully connected layer to output 4 classes (groups)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,  # Input features to the final layer
        out_features=4  # Number of output classes
    )

    # Move the model to the selected computation device
    model.to(device=device)

    # Define the optimizer for training the model
    optimizer = torch.optim.Adam(
        params=model.parameters(),  # Model parameters to optimize
        lr=1e-3  # Learning rate
    )

    # Define the loss function for classification
    loss_fcn = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch_no in itertools.count(start=1):  # Start counting epochs from 1
        print(f"*** Epoch {epoch_no} ***")  # Print the current epoch number

        for sample in data_loader:  # Iterate over batches in the DataLoader
            # Move images and labels to the selected device
            images = sample['image'].to(device=device)
            labels = sample['group'].to(device=device)

            # Perform a forward pass through the model
            outputs = model(images)

            # Compute the loss for this batch
            loss = loss_fcn(input=outputs, target=labels)

            # Perform backpropagation and optimization
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters

            # Print the loss for this batch
            print(f"Loss = {loss.item()}")

# Entry point of the script
if __name__ == "__main__":
    main()
