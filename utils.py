import random
import numpy as np


def split_file(file_name):
    # Open the file in read mode
    with open(file_name, "r") as file:
        # Read all the lines from the file and store them in a list
        lines = file.readlines()

    # Shuffle the order of the lines randomly
    random.shuffle(lines)

    # Determine the number of lines for training and testing
    train_lines = lines[:int(0.8 * len(lines))]  # Take the first 80% of lines for training
    test_lines = lines[int(0.8 * len(lines)):]  # Take the remaining 20% of lines for testing

    # Open the train file and write the training lines to it
    with open("train1.txt", "w") as file:
        file.writelines(train_lines)

    # Open the test file and write the testing lines to it
    with open("test1.txt", "w") as file:
        file.writelines(test_lines)


def load_data(file_name):
    data = []
    labels = []
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip().split()
            data.append(list(line[0]))
            labels.append(int(line[1]))
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    # Convert labels to one-hot encoding
    labels = np.expand_dims(labels, axis=1)
    return data, labels


def load_test(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip().split()
            data.append(list(line[0]))
    data = np.array(data, dtype=np.float32)
    return data


def clean_labels_from_file(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip().split()
            data.append(str(line[0]) + "\n")
    with open("testnet1.txt", "w") as file:
        file.writelines(data)


if __name__ == '__main__':
    # Call the split_file function with the given file name as an argument
    split_file("nn1.txt")
    # X_train, y_train = load_data("train0.txt")
    # X_test, y_test = load_data("test0.txt")
    clean_labels_from_file("nn1.txt")
