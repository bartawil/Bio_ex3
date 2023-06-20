import numpy as np
from genetic_neural_network import GeneticNeuralNetwork
from utils import load_data


def buildnet1(train_file, test_file):
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)

    # Create and train the neural network
    input_size = X_train.shape[1]
    hidden_size = 32
    output_size = 1

    # Initialize the population
    population_size = 100
    mutation_rate = 0.15

    neural_network = GeneticNeuralNetwork(population_size, mutation_rate, hidden_size, output_size)

    # Training parameters
    num_epochs = 3000
    num_parents = int(population_size / 2)

    W1, b1, W2, b2 = neural_network.train(input_size, X_train, y_train, num_epochs, num_parents)

    # Compute the final predictions
    y_pred_test = neural_network.predict(X_test, W1, b1, W2, b2)
    print("Test Accuracy:", (y_pred_test == y_test).mean())

    # Save the network architecture and weights
    with open('wnet1.txt', 'w') as file:
        # Write network architecture
        file.write(f"Input size: {input_size}\n")
        file.write(f"Hidden size: {hidden_size}\n")
        file.write(f"Output size: {output_size}\n")

        # Write weights and biases
        file.write("W1:\n")
        np.savetxt(file, W1, fmt='%f')
        file.write("b1:\n")
        np.savetxt(file, b1, fmt='%f')
        file.write("W2:\n")
        np.savetxt(file, W2, fmt='%f')
        file.write("b2:\n")
        np.savetxt(file, b2, fmt='%f')


if __name__ == '__main__':
    buildnet1("train1.txt", "test1.txt")
