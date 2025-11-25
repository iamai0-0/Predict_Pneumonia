from libs import *

def analyze_data(data_dir):
    """
    Analyze and visualize the distribution of classes in the training set.

    Parameters:
    - data_dir (str): Path to the main data directory.

    Returns:
    - dict: A dictionary containing the count of samples for each class.
    """
    train_dir = os.path.join(data_dir, 'train')  
    normal_train_dir = os.path.join(train_dir, 'NORMAL')  
    pneumonia_train_dir = os.path.join(train_dir, 'PNEUMONIA')  

    if not os.path.exists(normal_train_dir) or not os.path.exists(pneumonia_train_dir):
        raise FileNotFoundError("One or more specified data directories do not exist. Please check the paths.")

    # Count samples
    n_samples_nr_train = len(os.listdir(normal_train_dir))  
    n_samples_pn_train = len(os.listdir(pneumonia_train_dir))  

    # Define result
    class_count = {0: n_samples_nr_train, 1: n_samples_pn_train}
    class_names = {0: 'NORMAL', 1: 'PNEUMONIA'}


    print(f'Found {class_count[0]} elements for {class_names[0]}')
    print(f'Found {class_count[1]} elements for {class_names[1]}')

    # plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar([class_names[0], class_names[1]], [class_count[0], class_count[1]])
    ax.set_title('Class Distribution in Training Set')
    ax.set_ylabel('Number of Samples')
    ax.set_xlabel('Class')
    plt.show()

    return class_count


def class_weights(class_count):
    """
    Calculate class weights to handle imbalanced datasets.

    Parameters:
    - class_count (dict): A dictionary with class indices as keys and sample counts as values.

    Returns:
    - torch.FloatTensor: A tensor containing class weights.
    """
    samples_0 = class_count[0]
    samples_1 = class_count[1]
    tot_samples = samples_0 + samples_1

    # Calculate weights
    weight_0 = 1 - samples_0 / tot_samples
    weight_1 = 1 - weight_0  # equivalent to 1 - samples_1 / tot_samples

    # Create class weights tensor
    class_weights = [weight_0, weight_1]
    class_weights_tensor = torch.FloatTensor(class_weights)

    # Print weights
    print(f"Class weights: {class_weights}")
    return class_weights_tensor


if __name__ == "__main__":
    data_dir = "D:\GitHub\Processing_Image\Data" #Fill in " " your path

    # Analyze data distribution
    class_count = analyze_data(data_dir)

    # Calculate and print class weights
    class_weights = class_weights(class_count)