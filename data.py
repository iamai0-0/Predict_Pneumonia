from libs import *

def prepare_dataset(data_dir, sub_dir='train'):
    """
    Prepare the dataset with transformations and split it into training and validation sets.

    Parameters:
    - data_dir (str): Path to the main data directory.
    - sub_dir (str): The subdirectory where the train data is stored (default: 'train').

    Returns:
    - train_split: Training dataset split.
    - val_split: Validation dataset split.
    """

    full_dir = os.path.join(data_dir, sub_dir)
    if not os.path.exists(full_dir):
        raise FileNotFoundError(f"Directory {full_dir} doesn't exist. Please check the path again!")

    # Define transformations to apply to the data
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomGrayscale(),
        transforms.RandomAffine(translate=(0.05, 0.05), degrees=0),
        transforms.ToTensor()
    ])

    # Load the dataset using ImageFolder and apply transformations
    dataset = datasets.ImageFolder(full_dir, transform=data_transform)
    print(f"Classes: {dataset.classes}")
    print(f"Total images: {len(dataset)}")

    # Split dataset into training and validation sets
    train_split, val_split = train_test_split(dataset, test_size=0.3, random_state=42)
    print(f"Training set size: {len(train_split)}")
    print(f"Validation set size: {len(val_split)}")

    return train_split, val_split


def load_data(data_dir, sub_dir='train', batch_size=64):
    train_split, val_split = prepare_dataset(data_dir, sub_dir)

    # Create DataLoader for train and validation datasets
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=True)
    
    # Define class index mapping
    class_index = {0: 'NORMAL', 1: 'PNEUMONIA'}

    return train_loader, val_loader, class_index


def visualize_images(train_loader, class_index, n_rows=2, n_cols=5):
    """
    Visualizes a few images from the training data along with their class labels.

    Args:
        train_loader (DataLoader): The DataLoader object for the training data.
        class_index (dict): A dictionary mapping class indices to class names.
        n_rows (int): Number of rows for the grid of images.
        n_cols (int): Number of columns for the grid of images.
    """

    images, labels = next(iter(train_loader))
    plt.figure(figsize=(20, 10))

    # Loop over the grid positions and display images
    for i in range(n_rows * n_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))  
        plt.title(class_index[labels.numpy()[i]])  # Set title as the class label
        plt.axis('off')  

    plt.subplots_adjust(wspace=.02, hspace=-.2)
    plt.show()

if __name__ == "__main__":
    data_dir = "/content/drive/MyDrive/Processing_Image/Data"

    # Load data 
    train_loader, val_loader, class_index = load_data(data_dir)

    # Visualize 
    visualize_images(train_loader, class_index)
