import numpy as np


class GeneticNeuralNetwork:
    def __init__(self, population_size, mutation_rate, hidden_size=32, output_size=1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.population = None

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, X, W1, b1, W2, b2):
        # Layer 1 (Input to Hidden)
        z1 = np.dot(X, W1) + b1
        a1 = self.sigmoid(z1)

        # Layer 2 (Hidden to Output)
        z2 = np.dot(a1, W2) + b2
        a2 = self.sigmoid(z2)

        return a1, a2

    @staticmethod
    def compute_loss(y, y_pred):
        return np.mean(-(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))

    def predict(self, X, W1, b1, W2, b2):
        _, y_pred = self.forward_propagation(X, W1, b1, W2, b2)
        return np.round(y_pred)

    def initialize_population(self, input_size):
        self.population = []
        for _ in range(self.population_size - 1):
            W1 = np.random.randn(input_size, self.hidden_size)
            b1 = np.zeros(self.hidden_size)
            W2 = np.random.randn(self.hidden_size, self.output_size)
            b2 = np.zeros(self.output_size)
            individual = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
            self.population.append(individual)

    def evaluate_population(self, X, y):
        scores = []
        for individual in self.population:
            W1, b1, W2, b2 = individual['W1'], individual['b1'], individual['W2'], individual['b2']
            _, y_pred = self.forward_propagation(X, W1, b1, W2, b2)
            loss = self.compute_loss(y, y_pred)
            scores.append(loss)
        return scores

    @staticmethod
    def select_parents(scores, num_parents):
        parents_indices = np.argsort(scores)[:num_parents]
        return parents_indices

    def crossover(self, parents_indices, input_size):
        offspring = []
        for _ in range(self.population_size):
            parent1, parent2 = np.random.choice(parents_indices, size=2, replace=False)
            parent1 = self.population[parent1]
            parent2 = self.population[parent2]

            # Perform crossover
            child = {'W1': np.copy(parent1['W1']), 'b1': np.copy(parent1['b1']),
                     'W2': np.copy(parent2['W2']), 'b2': np.copy(parent2['b2'])}

            # Perform mutation
            for weight in ['W1', 'b1', 'W2', 'b2']:
                if np.random.rand() < self.mutation_rate:
                    child[weight] += np.random.randn(*child[weight].shape)

            offspring.append(child)

        return offspring

    @staticmethod
    def update_population(population, offspring):
        new_population = population[:len(offspring)] + offspring
        return new_population

    def train(self, input_size, X_train, y_train, num_epochs, num_parents):
        self.initialize_population(input_size)
        for epoch in range(num_epochs):
            # Evaluate the population
            scores = self.evaluate_population(X_train, y_train)

            # Select parents for reproduction
            parents_indices = self.select_parents(scores, num_parents)

            # Perform crossover and mutation
            offspring = self.crossover(parents_indices, input_size)

            # Update the population with the new offspring
            self.population = self.update_population(self.population, offspring)

            # Display the best individual's loss
            best_individual_index = np.argmin(scores)
            loss = scores[best_individual_index]
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        # Select the best individual
        best_individual_index = np.argmin(scores)
        best_individual = self.population[best_individual_index]
        W1, b1, W2, b2 = best_individual['W1'], best_individual['b1'], best_individual['W2'], best_individual['b2']

        return W1, b1, W2, b2

    def test_accuracy(self, X_test, y_test, W1, b1, W2, b2):
        predictions = self.predict(X_test, W1, b1, W2, b2)
        accuracy = (predictions == y_test).mean()
        return accuracy
