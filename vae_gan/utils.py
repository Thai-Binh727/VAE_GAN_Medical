from torchvision.utils import make_grid, save_image
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def show_and_save(folder_name, file_name, img):
    # Delay the import of anything that might cause a circular import
    npimg = np.transpose(img.numpy(), (1, 2, 0))

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Create the full path for saving the image
    file_path = os.path.join(folder_name, f"{file_name}.jpg")

    # Create a figure with high resolution
    fig = plt.figure(dpi=200)
    fig.suptitle(file_name, fontsize=14, fontweight='bold')

    # Display the image
    plt.imshow(npimg)

    # Save the image in the specified folder
    plt.imsave(file_path, npimg)


def plot_loss(loss_list):
    plt.figure(figsize=(10, 5))
    plt.title("Loss During Training")
    plt.plot(loss_list, label="Loss")

    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()