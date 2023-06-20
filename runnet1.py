import numpy as np
from genetic_neural_network import GeneticNeuralNetwork
from utils import load_test


def runnet1(wnet_file, data_file):
    with open(wnet_file, 'r') as file:
        lines = file.readlines()

        input_size = int(lines[0].split(':')[1].strip())
        hidden_size = int(lines[1].split(':')[1].strip())
        output_size = int(lines[2].split(':')[1].strip())

        # Load weights and biases
        W1 = np.loadtxt(lines[4:4 + input_size], dtype=np.float32)
        b1 = np.loadtxt(lines[5 + input_size:5 + input_size + hidden_size], dtype=np.float32)
        W2 = np.loadtxt(lines[6 + input_size + hidden_size:6 + input_size + hidden_size + hidden_size],
                        dtype=np.float32)
        b2 = np.loadtxt(lines[7 + input_size + hidden_size + hidden_size:], dtype=np.float32)

        # Reshape W2 and b2
        W2 = np.reshape(W2, (hidden_size, 1))
        b2 = np.reshape(b2, (1,))

    neural_network = GeneticNeuralNetwork(input_size, hidden_size, output_size)
    X_test = load_test(data_file)
    output = neural_network.predict(X_test, W1, b1, W2, b2)
    with open("output1.txt", "w") as file:
        for element in output:
            file.write(str(int(element[0])) + "\n")


if __name__ == '__main__':
    runnet1("wnet1.txt", "testnet1.txt")