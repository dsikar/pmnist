import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the path
sys.path.append(parent_dir)
# NB a symlink to ../work-in-progress/utils is also required
# ln -s ~/git/work-in-progress/utils/ utils
from utils.distance_metrics import DistanceMetric
from utils.perturbations import *
from utils.helper_functions import *
from utils.perturbation_levels import PERTURBATION_LEVELS

def read_bytes_from_file(file_path, num_bytes, start=0, verbose=False):
    """
    Reads a specified number of bytes from a file starting at a given offset and prints them in hexadecimal.

    Parameters:
    - file_path: The path to the file.
    - num_bytes: The number of bytes to read from the file.
    - start: The offset from the beginning of the file to start reading bytes (default is 0).
    """
    with open(file_path, 'rb') as file:
        file.seek(start)  # Move to the start position
        data = file.read(num_bytes)
        if verbose:
          print("Hexadecimal representation of", num_bytes, "bytes starting from byte", start, ":")
          print(data.hex())
    return data.hex()

def hex_to_numpy_array(hex_data, row_length):
    """
    Converts a hexadecimal string into a numpy array of specified row length.

    Parameters:
    - hex_data: Hexadecimal string to be converted.
    - row_length: The length of each row in the resulting array.

    Returns:
    - A numpy array representing the hexadecimal data.

    Example:
    # Assuming read_bytes_from_file has been called and hex_data is obtained
    file_path = 'data/MNIST/raw/train-images-idx3-ubyte'
    num_bytes = 784
    start=16
    verbose = False
    image1 = read_bytes_from_file(file_path, num_bytes, start, verbose)
    hex_data = image1  # This is a placeholder. Use actual hex data from read_bytes_from_file
    row_length = 28  # For MNIST images

    try:
        image_array = hex_to_numpy_array(hex_data, row_length)
        print("Numpy array shape:", image_array.shape)
    except ValueError as e:
        print(e)
    """
    # Convert hex_data to bytes in decimal format
    byte_data = bytes.fromhex(hex_data)

    # Calculate the total number of expected rows
    total_bytes = len(byte_data)
    if total_bytes % row_length != 0:
        raise ValueError("The total number of bytes is not evenly divisible by the specified row length.")

    # Calculate the number of rows
    num_rows = total_bytes // row_length

    # Convert byte data to a numpy array and reshape
    np_array = np.frombuffer(byte_data, dtype=np.uint8).reshape((num_rows, row_length))

    return np_array

def display_image(image_array):
    """
    Displays an image from a numpy array.

    Parameters:
    - image_array: A numpy array representing the image to be displayed.

    Example:
    file_path = 'data/MNIST/raw/train-images-idx3-ubyte'
    num_bytes = 784
    start=16+num_bytes
    verbose = False
    image2 = read_bytes_from_file(file_path, num_bytes, start, verbose)
    hex_data = image2  # This is a placeholder. Use actual hex data from read_bytes_from_file
    row_length = 28
    image_array = hex_to_numpy_array(hex_data, row_length)
    display_image(image_array)
    """
    plt.imshow(image_array, cmap='gray')
    plt.colorbar()
    plt.show()

def display_image_with_histogram(image_array):
    """
    Displays an image from a numpy array and a histogram of its pixel values.

    Parameters:
    - image_array: A numpy array representing the image to be displayed.

    Example:
    file_path = 'data/MNIST/raw/train-images-idx3-ubyte'
    num_bytes = 784
    start=16+num_bytes
    verbose = False
    image2 = read_bytes_from_file(file_path, num_bytes, start, verbose)
    hex_data = image2  # This is a placeholder. Use actual hex data from read_bytes_from_file
    row_length = 28
    image_array = hex_to_numpy_array(hex_data, row_length)
    display_image_with_histogram(image_array)
    """
    # Create a figure with 1 row and 2 columns of subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display the image
    ax[0].imshow(image_array, cmap='gray')
    ax[0].set_title('Image')
    ax[0].axis('off')  # Hide axis ticks and labels

    # Display the histogram
    ax[1].hist(image_array.ravel(), bins=50, color='gray')
    ax[1].set_title('Pixel Value Distribution')
    ax[1].set_xlabel('Pixel Intensity')
    ax[1].set_ylabel('Frequency')

    # Show the plots
    plt.tight_layout()  # Adjust the layout to make room for the titles
    plt.show()

def display_mnist_lbl(filename, index, verbose = False):
    """
    Display the label at a given index

    Parameters
    ==========
    filename: string, the mnist binary file
    index: the index to display
    verbose: boolean, display debug info

    Example
    =========
    pmnist_lbl = 'data/MNIST/raw/train-labels-idx1-ubyte' # 'perturbed-train-labels-idx1-ubyte'
    index = 0
    verbose = True
    display_mnist_lbl(pmnist_lbl, index, verbose)
    Hexadecimal representation of 1 bytes starting from byte 8 :
    05
    In file data/MNIST/raw/train-labels-idx1-ubyte, label index 0 is 05
    """
    num_bytes = 1
    start=8+index
    # Check the label index is within bounds
    file_size = os.path.getsize(filename)
    if file_size < start+num_bytes:
        print("The specified label index {} is out of bounds for the file.".format(index))
    else:
        lbl = read_bytes_from_file(filename, num_bytes, start, verbose)
        print("In file {}, label index {} is {}".format(filename, index, lbl))

def display_mnist_img(filename, index, verbose = False):
    """
    Display image and histogram at a given index

    Parameters
    ==========
    filename: string, the mnist binary file
    index: the index to display
    verbose: boolean, display debug info

    Example
    =========
    pmnist_img = 'data/MNIST/raw/train-images-idx3-ubyte'  # perturbed-train-images-idx3-ubyte
    index = 0
    verbose = False
    display_mnist_img(pmnist_img, index, verbose)
    """
    num_bytes = 784 #(28x28)
    start=16+(index*num_bytes)
    row_length = 28
    file_size = os.path.getsize(filename)
    if file_size < start+num_bytes:
        print("The specified image index {} is out of bounds for the file.".format(index))
    else: 
        img_hex = read_bytes_from_file(filename, num_bytes, start, verbose)
        image_array = hex_to_numpy_array(img_hex, row_length)
        display_image_with_histogram(image_array)

def display_pmnist_perturbation(filename, index, verbose = True):
  """
  Display image perturbation and level if applicable

  Parameters
  ==========
  filename: string, the mnist binary file
  index: the index to display
  verbose: boolean, display debug info

  Example
  =========
  pmnist_perturbations = 'data/MNIST/raw/train-labels-idx1-ubyte'  # 'perturbation-train-levels-idx0-ubyte'
  index = 0
  verbose = True
  display_pmnist_perturbation(pmnist_perturbations, index, verbose)
  """
  num_bytes = 1
  key_start = 8 + (index * 2)
  level_start = 8 + (index * 2) + 1
  # POS.....8.....9.......10....11......12....13
  # HEADER..KEY0..LEVEL0..KEY1..LEVEL1..KEY2..LEVEL2
  # HEADER LENGTH = 8
  # TO RETRIEVE KEY0
  # index = 0
  # START = 8 + index (8)
  # TO RETRIEVE LEVEL0
  # START = 8 + index + 1 (9)
  # TO RETRIEVE KEY1
  # index = 1
  # START = 8 + (index*2) = 10
  # TO RETRIEVE LEVEL1
  # START = 8 + (index*2) + 1 (11)
  # TO RETRIEVE KEY2
  # index = 2
  # START = 8 + (index*2) = 12
  # TO RETRIEVE LEVEL2
  # START = 8 + (index*2) + 1 (13)
  key = read_bytes_from_file(filename, num_bytes, key_start, verbose)
  key_index = int(key, 16)
  if key_index == 255: # 0xFF, clean image
    print("Original image, not perturbed.")
    return
  key = get_key_by_index(key_index)
  level = read_bytes_from_file(filename, num_bytes, level_start, verbose)
  print("In file: {}, for index: {}, key index: {}, key is {}, level is {}".format(filename, index, key_index, key, level))

def random_perturbation(img):
  """
  Apply random perturbation to an image

  Parameters
  ==========
  img: numpy image array
  Returns
  ==========
  p_img: numpy array, perturbed image
  key: string, perturbation key
  level: int, perturbation level

  Notes
  ==========
  Assume repo is cloned and imports have been made outside prior to function being called
  # git clone https://github.com/dsikar/work-in-progress.git work_in_progress
  import sys
  import os
  subdirectory_path = 'work_in_progress/utils'
  sys.path.append(subdirectory_path)

  from work_in_progress.utils.perturbations import *
  from work_in_progress.utils.helper_functions import *
  from work_in_progress.utils.perturbation_levels import PERTURBATION_LEVELS
  """
  import random

  pt = Perturbation(pixel_range=(0, 255))
  key = random.choice(list(PERTURBATION_LEVELS.keys()))
  level = random.randint(0, 9)
  kwargs = PERTURBATION_LEVELS[key][level]
  p_img = getattr(pt, key)(img, **kwargs)
  # Convert the perturbed image to np.uint8 so it can be written to a binary file
  p_img_uint8 = np.array(p_img, dtype=np.uint8)
  return p_img_uint8, key, level

def find_perturbation_key_index(key_to_find):
  """
  Find the perturbation key index
  Parameters
  =========
  key_to_find: string
  Returns

  =========
  key_index: integet

  Note
  =========
  Assume repo is cloned and imports have been made outside prior to function being called
  # git clone https://github.com/dsikar/work-in-progress.git work_in_progress
  import sys
  import os
  subdirectory_path = 'work_in_progress/utils'
  sys.path.append(subdirectory_path)

  from work_in_progress.utils.perturbation_levels import PERTURBATION_LEVELS
  """
  keys_list = list(PERTURBATION_LEVELS.keys())
  key_index = keys_list.index(key_to_find)
  return key_index

def get_key_by_index(index=0, dictionary=PERTURBATION_LEVELS, ):
    """
    Retrieves the key from a dictionary given its index.

    Parameters:
    - index: The index of the key to retrieve.
    - dictionary: The dictionary from which to retrieve the key.

    Returns:
    - The key at the specified index, or None if the index is out of bounds.

    Example:
    key = get_key_by_index(1)
    print(key)
    """
    if index < 0 or index >= len(dictionary):
        return None
    return list(dictionary.keys())[index]

def append_bytes_to_file(filename, value, num_bytes):
    """
    Appends a specified value as bytes to a file.

    Parameters:
    - filename: The name of the file to append the bytes to.
    - value: The integer value to be appended as bytes.
    - num_bytes: The number of bytes to represent the value.
    """
    # print("num_bytes: {}".format(num_bytes))
    with open(filename, 'ab') as file:  # Open the file in append binary mode
        file.write(value.to_bytes(num_bytes, 'big'))

def append_single_image_to_file(filename, image):
    """
    Appends a single image's pixel data to the specified file.

    Parameters:
    - filename: The name of the file to append the image to.
    - image: A numpy array representing the image. The array should be 28x28 and of dtype np.uint8.
    """
    import numpy as np

    if image.shape != (28, 28):
        raise ValueError("Image must be 28x28 pixels.")
    if image.dtype != np.uint8:
        raise ValueError("Image data must be of type np.uint8")

    with open(filename, 'ab') as file:  # Open the file in append binary model
        file.write(image.tobytes())

# remove and create file
def create_file(filename, verbose = False):

  # Check if the file exists and remove it
  if os.path.exists(filename):
      os.remove(filename)
      if verbose:
        print(f"File {filename} has been removed.")
  else:
      if verbose:
        print(f"File {filename} does not exist.")

  # create
  with open(filename, 'a'):
      if verbose:
        print(f"File {filename} has been created or already exists.")

  # Verify creation
  if os.path.exists(filename):
      if verbose:
        print(f"Verification: File {filename} exists.")
  else:
      if verbose:
        print(f"Verification failed: File {filename} does not exist.")

def create_mnist_data_file(image_filename, file_count):
  # image_filename = 'perturbed-train-images-idx3-ubyte'
  create_file(image_filename)

  magic_number = 0x00000803
  num_images = 120000 # Double the 60000, that is, clear plus noisy
  rows = 28
  cols = 28
  num_bytes = 4

  # magic number
  append_bytes_to_file(image_filename, magic_number, num_bytes)
  # num_images
  append_bytes_to_file(image_filename, num_images, num_bytes)
  # rows
  append_bytes_to_file(image_filename, rows, num_bytes)
  # columns
  append_bytes_to_file(image_filename, cols, num_bytes)

  # sanity check
  print("Sanity Check")
  num_bytes = 400
  start=0
  verbose = True
  read_bytes_from_file(image_filename, num_bytes, start, verbose)

def create_mnist_file(filename, header_info, verbose = False):
    """
    Writes given values to a binary file, with each value written according to its specified byte size.

    Parameters:
    - filename: The name of the file to write to.
    - values_dict: A dictionary where each key-value pair is the value to write and the number of bytes to use.

    Example:
    perturbed_train_filename = 'perturbed-train-images-idx3-ubyte'
    header_info = [
        (0x00000803, 4),  # Magic number for images, 4 bytes
        (120000, 4),      # Number of images or labels, 4 bytes
        (28, 4),         # Rows, 4 bytes (only for image files)
        (28, 4)          # Columns, 4 bytes (only for image files)
    ]

    create_mnist_file(perturbed_train_filename, header_info, verbose = True)
    """
    # reset if existing
    create_file(filename, verbose)
    for value, num_bytes in header_info:
        # file.write(value.to_bytes(num_bytes, 'big'))
        append_bytes_to_file(filename, value, num_bytes)

    # sanity check
    if verbose:
        print("Sanity Check")
        num_bytes = 400
        start=0
        #verbose = True
        read_bytes_from_file(filename, num_bytes, start, verbose)

def generate_perturbations(img):
    """
    Generate all possible perturbations of an image.

    Parameters
    ==========
    img: numpy image array

    Returns
    ==========
    perturbations: list of tuples, each containing (perturbed image, perturbation key, perturbation level)

    Notes
    ==========
    Assume repo is cloned and imports have been made outside prior to function being called
    # git clone https://github.com/dsikar/work-in-progress.git work_in_progress
    import sys
    import os
    subdirectory_path = 'work_in_progress/utils'
    sys.path.append(subdirectory_path)
    from work_in_progress.utils.perturbations import *
    from work_in_progress.utils.helper_functions import *
    from work_in_progress.utils.perturbation_levels import PERTURBATION_LEVELS
    """
    pt = Perturbation()
    perturbations = []

    for key in PERTURBATION_LEVELS.keys():
        for level in range(len(PERTURBATION_LEVELS[key])):
            kwargs = PERTURBATION_LEVELS[key][level]
            # scale img to -1 +1, as per work-in-progress/utils/mnist_cnn_eval.py
            img_copy = img / 255.0 * 2.0 - 1.0
            p_img = getattr(pt, key)(img_copy, **kwargs)
            # scale p_img back to 0-255
            p_img = (p_img * 0.5 + 0.5) * 255.0
            p_img_uint8 = np.array(p_img, dtype=np.uint8)
            perturbations.append((p_img_uint8, key, level))

    return perturbations        
        
    #     for level in range(10):
    #         if level in PERTURBATION_LEVELS[key]:
    #             kwargs = PERTURBATION_LEVELS[key][level]
    #             p_img = getattr(pt, key)(img, **kwargs)
    #             p_img_uint8 = np.array(p_img, dtype=np.uint8)
    #             perturbations.append((p_img_uint8, key, level))

    # return perturbations        

def gen_pmnist_dataset_all_possibilities(img_path, lbl_path, pmnist_img, pmnist_lbl, pmnist_perturbations, num_files=1, verbose=False):
    """
    Generate the PMNIST dataset for every possible perturbation

    Parameters
    ==========
    img_path: string, path to the MNIST image file
    lbl_path: string, path to the MNIST label file
    pmnist_img: string, path to the PMNIST image file
    pmnist_lbl: string, path to the PMNIST label file
    pmnist_perturbations: string, path to the PMNIST perturbation file
    num_files: int, number of files to process
    verbose: boolean, print debug info

    Note
    =========
    The maximum number of files for the training dataset is 60000, 10000 for the testing dataset

    Example
    =========
    img_path = 'data/MNIST/raw/train-images-idx3-ubyte'
    lbl_path = 'data/MNIST/raw/train-labels-idx1-ubyte'
    pmnist_img = 'perturbed-train-images-idx3-ubyte'
    pmnist_lbl = 'perturbed-train-labels-idx1-ubyte'
    pmnist_perturbations = 'perturbation-train-levels-idx0-ubyte'
    num_files = 1
    verbose = True
    gen_pmnist_dataset(img_path, lbl_path, pmnist_img, pmnist_lbl, pmnist_perturbations, num_files, verbose)
    """
    # reset files
    # init_perturbed_mnist_files(verbose)

    # Number of files in the MNIST training dataset
    num_img_files = num_files  # 60000

    # paths
    mnist_image_file_path = img_path  # 'data/MNIST/raw/train-images-idx3-ubyte'
    mnist_label_file_path = lbl_path  # 'data/MNIST/raw/train-labels-idx1-ubyte'

    # image and label data size in bytes
    num_img_bytes = 784  # (28x28)
    num_label_bytes = 1

    # MNIST header lengths
    img_header_len = 16
    label_header_len = 8

    # row x col lengths
    row_length = 28

    # The big for loop
    for i in range(0, num_img_files):
        # offsets
        img_offset = img_header_len + num_img_bytes * i
        label_offset = label_header_len + num_label_bytes * i
        # use the original label
        label = read_bytes_from_file(mnist_label_file_path, num_label_bytes, label_offset, verbose)
        # print("label:", label)
        # convert hex to int
        label = int(label, 16)
        img_hex = read_bytes_from_file(mnist_image_file_path, num_img_bytes, img_offset, verbose)
        # convert to array
        img_array = hex_to_numpy_array(img_hex, row_length)
        if verbose:
            # sanity check, display label and image
            print("label: {}".format(label))
            # display original image
            display_image_with_histogram(img_array)

        # Process and save image and metadata
        # 1. save the original image to perturbed image dataset
        perturbed_train_images_idx3_ubyte = pmnist_img  # 'perturbed-train-images-idx3-ubyte'
        append_single_image_to_file(perturbed_train_images_idx3_ubyte, img_array)
        if verbose:
            print("Saved original image to perturbed image dataset: {}".format(perturbed_train_images_idx3_ubyte))
        # 2. save label to the perturbed label dataset
        perturbed_train_labels_idx1_ubyte = pmnist_lbl  # 'perturbed-train-labels-idx1-ubyte'
        num_bytes = 1
        append_bytes_to_file(perturbed_train_labels_idx1_ubyte, label, num_bytes)
        if verbose:
            print("Saved 'clean' label to perturbed image dataset: {}".format(perturbed_train_labels_idx1_ubyte))        
        # 5. save the perturbation type and label for the original image to the perturbation training levels dataset
        perturbation_train_levels_idx0_ubyte = pmnist_perturbations  # 'perturbation-train-levels-idx0-ubyte'
        val = 0xFFFF # NB 255, 255 indicates the original image.
        num_bytes = 2
        append_bytes_to_file(perturbation_train_levels_idx0_ubyte, val, num_bytes)
        if verbose:
            print("Saved key and level to perturbed levels dataset")

        # generate all perturbations for the image
        perturbations = generate_perturbations(img_array)
        for p_img, key, level in perturbations: 
            key_index = find_perturbation_key_index(key)
            if verbose:
                # display perturbation parameters
                print("Perturbation: {}, Perturbation index: {}, level: {}".format(key, key_index, level))
                # display perturbed image
                display_image_with_histogram(p_img)

            # Process and save image and metadata
            # 3. save the perturbed image to the perturbed dataset
            append_single_image_to_file(perturbed_train_images_idx3_ubyte, p_img)
            if verbose:
                print("Saved perturbed image to perturbed image dataset: {}".format(perturbed_train_images_idx3_ubyte))
            # 4. save the label to the perturbed label dataset (again)
            append_bytes_to_file(perturbed_train_labels_idx1_ubyte, label, num_bytes)    
            if verbose:
                print("Saved label to perturbed labels dataset: {}".format(perturbed_train_labels_idx1_ubyte))
            # 6. save the perturbation type and level for the perturbed image
            num_bytes = 1
            append_bytes_to_file(perturbation_train_levels_idx0_ubyte, key_index, num_bytes)
            append_bytes_to_file(perturbation_train_levels_idx0_ubyte, level, num_bytes)
            if verbose:
                print("Saved key and level to perturbed levels dataset")
        if verbose:
            print("Processed image: {}".format(i + 1))

    # Sanity check
    if verbose:
        for index in range(0, num_img_files):
            # show label i
            display_mnist_lbl(pmnist_lbl, index)
            # show image i
            display_mnist_img(pmnist_img, index)
            # show perturbation i
            display_pmnist_perturbation(pmnist_perturbations, index)

def init_perturbed_mnist_files(
        perturbed_train_images_file='perturbed-train-images-idx3-ubyte',
        perturbed_test_images_file='t20k-perturbed-images-idx3-ubyte',
        perturbed_train_labels_file='perturbed-train-labels-idx1-ubyte',
        perturbed_test_labels_file='t20k-perturbed-labels-idx1-ubyte',
        perturbation_train_levels_file='perturbation-train-levels-idx0-ubyte',
        perturbation_test_levels_file='t20k-perturbation-levels-idx0-ubyte',
        num_train_files=120000,
        num_test_files=20000,
        verbose=True):
    """
    Create the set of perturbed mnist files

    Parameters
    ==========
    perturbed_train_images_file: str, name of the perturbed training images file (default: 'perturbed-train-images-idx3-ubyte')
    perturbed_test_images_file: str, name of the perturbed testing images file (default: 't20k-perturbed-images-idx3-ubyte')
    perturbed_train_labels_file: str, name of the perturbed training labels file (default: 'perturbed-train-labels-idx1-ubyte')
    perturbed_test_labels_file: str, name of the perturbed testing labels file (default: 't20k-perturbed-labels-idx1-ubyte')
    perturbation_train_levels_file: str, name of the perturbation training levels file (default: 'perturbation-train-levels-idx0-ubyte')
    perturbation_test_levels_file: str, name of the perturbation testing levels file (default: 't20k-perturbation-levels-idx0-ubyte')
    num_train_files: int, number of training files (default: 120000)
    num_test_files: int, number of testing files (default: 20000)
    verbose: boolean, print debug info (default: True)
    """
    # 1. perturbed-train-images-idx3-ubyte
    header_info = [
        (0x00000803, 4),  # Magic number for images, 4 bytes
        (num_train_files, 4),  # Number of images or labels, 4 bytes
        (28, 4),  # Rows, 4 bytes (only for image files)
        (28, 4)  # Columns, 4 bytes (only for image files)
    ]
    create_mnist_file(perturbed_train_images_file, header_info, verbose)
    print("====================================")

    # 4. t20k-perturbed-images-idx3-ubyte
    header_info = [
        (0x00000803, 4),
        (num_test_files, 4),
        (28, 4),
        (28, 4)
    ]
    create_mnist_file(perturbed_test_images_file, header_info, verbose)
    print("====================================")

    # 2. perturbed-train-labels-idx1-ubyte
    header_info = [
        (0x00000801, 4),  # Magic number for labels
        (num_train_files, 4)
    ]
    create_mnist_file(perturbed_train_labels_file, header_info, verbose)
    print("====================================")

    # 5. t20k-perturbed-labels-idx1-ubyte
    header_info = [
        (0x00000801, 4),
        (num_test_files, 4)
    ]
    create_mnist_file(perturbed_test_labels_file, header_info, verbose)
    print("====================================")

    # 3. perturbation-train-levels-idx0-ubyte
    header_info = [
        (0x000007FF, 4),  # Magic number for perturbation types and intensity levels
        (num_train_files, 4)
    ]
    create_mnist_file(perturbation_train_levels_file, header_info, verbose)
    print("====================================")

    # 6. t20k-perturbation-levels-idx0-ubyte
    header_info = [
        (0x000007FF, 4),  # Magic number for perturbation types and intensity levels
        (num_test_files, 4)
    ]
    create_mnist_file(perturbation_test_levels_file, header_info, verbose)
    print("====================================")