import os
import importlib
import sys
from pmnist_helper_functions import *

##########################
# Download MNIST dataset #
##########################
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download the MNIST dataset
train_set = datasets.MNIST(root='./data', download=False, train=True, transform=transform)
test_set = datasets.MNIST(root='./data', download=False, train=False, transform=transform)

####################
# Helper functions #
####################

# create files
init_perturbed_mnist_files(
        perturbed_train_images_file='perturbed-train-images-idx3-ubyte',
        perturbed_test_images_file='t1210k-perturbed-images-idx3-ubyte',
        perturbed_train_labels_file='perturbed-train-labels-idx1-ubyte',
        perturbed_test_labels_file='t1210k-perturbed-labels-idx1-ubyte',
        perturbation_train_levels_file='perturbation-train-levels-idx0-ubyte',
        perturbation_test_levels_file='t1210k-perturbation-levels-idx0-ubyte',
        num_train_files=7260000,
        num_test_files=1210000,
        verbose=True)

# process training dataset # STOPPED HERE 0. Change perturbation type and level to a byte value - DONE
#                                         STOPPED HERE 1. process PMNIST testing dataset saving sofmax values label, prediction, pertubation type and level
#                                         2. Create perturbations for CIFAR10
#                                         3. Create perturbed CIFAR10 dataset
#                                         4. process CIFAR-10 dataset saving sofmax values label, prediction, pertubation type and level
#                                         5. Add KL, BD and HI to PMNISt and CIFAR-10 predictions
#                                         6. Create 2D plots for PMNIST and CIFAR-10
img_path = 'data/MNIST/raw/train-images-idx3-ubyte'
lbl_path = 'data/MNIST/raw/train-labels-idx1-ubyte'
pmnist_img = 'perturbed-train-images-idx3-ubyte'
pmnist_lbl = 'perturbed-train-labels-idx1-ubyte'
pmnist_perturbations = 'perturbation-train-levels-idx0-ubyte'
num_files = 60000
verbose = False
gen_pmnist_dataset_all_possibilities(img_path, lbl_path, pmnist_img, pmnist_lbl, pmnist_perturbations, num_files, verbose)

# Only generate testing dataset perturbations
# process testing dataset
img_path = 'data/MNIST/raw/t10k-images-idx3-ubyte'
lbl_path = 'data/MNIST/raw/t10k-labels-idx1-ubyte'
pmnist_img = 't1210k-perturbed-images-idx3-ubyte'
pmnist_lbl = 't1210k-perturbed-labels-idx1-ubyte'
pmnist_perturbations = 't1210k-perturbation-levels-idx0-ubyte'
num_files = 10000
verbose = False
gen_pmnist_dataset_all_possibilities(img_path, lbl_path, pmnist_img, pmnist_lbl, pmnist_perturbations, num_files, verbose)
