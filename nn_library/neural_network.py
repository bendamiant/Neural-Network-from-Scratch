import numpy as np
from abc import ABC, abstractmethod



class NeuralNetwork(ABC):
    
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
               
    def train(self, algorithm, epochs, learning_rate, training_set, validation_set, cost_function, learning_rate_decay_rate=0, augment_images=False, augmented_fraction=0.1, mini_batch_size=None, save_best_parameters=False, seed=None):
        algorithm = algorithm.lower()
        cost_function = COST_FUNCTIONS.get(cost_function.lower())
        if seed:
            np.random.seed(seed)
            
        if augment_images:
            training_set = augment_training_images(training_set, augmented_fraction)
            
        if algorithm == "sgd":
            self._train_sgd(epochs, learning_rate, training_set, validation_set, mini_batch_size, cost_function, learning_rate_decay_rate, save_best_parameters, seed)
        elif algorithm == "bgd":
            self._train_bgd(epochs, learning_rate, training_set, validation_set, cost_function, learning_rate_decay_rate, save_best_parameters)
        else:
            raise ValueError(f"unknown algorithm: {algorithm}")
    
    def get_layer(self, index):
        return self.layers[index]
    
    @abstractmethod
    def _forward_prop(self):
        raise NotImplementedError()
    
    @abstractmethod
    def _back_prop(self):
        raise NotImplementedError()
        
    @abstractmethod
    def _train_sgd(self):
        raise NotImplementedError() 
        
    @abstractmethod
    def _train_bgd(self):
        raise NotImplementedError() 




class MultiLayerPerceptron(NeuralNetwork):
    
    def __init__(self, layers, weight_init="standard_normal", bias_init="standard_normal", constant=None):
        super().__init__(layers)
        self.best_params = None
        for i in range(1, self.num_layers):
            layer = self.layers[i]
            prev_layer = self.layers[i-1]
#             layer.biases = bias_init_function(bias_init, constant)(layer.units)
#             layer.weights = weight_init_function(weight_init)(layer.units, prev_layer.units)
            layer.set_biases(bias_init_function(bias_init, constant)(layer.units))
            layer.set_weights(weight_init_function(weight_init)(layer.units, prev_layer.units))
    
    def get_params(self):
        weights = []
        biases = []
        for layer in self.layers[1:]:
            weights.append(layer.weights)
            biases.append(layer.biases)
        return weights, biases
    
    def set_params(self, weights, biases):
        for i in range(1, self.num_layers):
            layer = self.layers[i]
            layer.set_biases = biases[i-1]
            layer.set_weights = weights[i-1]
    
    def set_best_params(self, weights, biases):
        self.best_params = (weights, biases)
    
    def _forward_prop(self, x):
        activations = []
        weighted_inputs = []  
        input_layer = self.get_layer(0)
        a = input_layer.layer_activation(x)
        activations.append(a) 
        for layer in self.layers[1:]:
            z, a = layer.layer_activation(a)
            activations.append(a)
            weighted_inputs.append(z)
        return weighted_inputs, activations
    
    def predict(self, x):
        weighted_inputs, activations = self._forward_prop(x)
        return activations[-1]
    
    def classify(self, x):
        prediction = self.predict(x)
        classification = np.zeros(prediction.shape)
        one_index = np.argmax(prediction, axis=0)
        classification[one_index] = np.array([1.0])
        return classification
    
    def _train_bgd(self, epochs, learning_rate, training_set, validation_set, cost_function, learning_rate_decay_rate, save_best_parameters, seed=None):
        best_weights = None
        best_biases = None
        best_valid_accuracy = -1.0
        for epoch in range(epochs):
            
            learning_rate = (1 / (1 + learning_rate_decay_rate * epoch)) * learning_rate
            
            self.update_mini_batch(training_set, cost_function, learning_rate)
            
            # Evaluate performance for this epoch
            train_cost = sum([cost_function(self.predict(x), y) for x, y in training_set]) / len(training_set)
            valid_cost = sum([cost_function(self.predict(x), y) for x, y in validation_set]) / len(validation_set)
            print(f"epoch {epoch} complete! | train_cost: {np.around(train_cost, 5)} | train_accuracy: {np.around(self.accuracy(training_set), 5)} | valid_cost: {np.around(valid_cost, 5)} | valid_accuracy: {np.around(self.accuracy(validation_set), 5)}\n")
            
            if save_best_parameters:
                if valid_accuracy > best_valid_accuracy:
                    best_weights, best_biases = self.get_params()
                    self.set_best_params(best_weights, best_biases)
                    
    def _train_sgd(self, epochs, learning_rate, training_set, validation_set, mini_batch_size, cost_function, learning_rate_decay_rate, save_best_parameters, seed=None):
        best_weights = None
        best_biases = None
        best_valid_accuracy = -1.0
        for epoch in range(epochs):
            if seed:
                np.random.seed(seed + epoch) 
            mini_batches = self.make_mini_batches(training_set, mini_batch_size)
            
            learning_rate = (1 / (1 + learning_rate_decay_rate * epoch)) * learning_rate
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, cost_function, learning_rate)
            
            # Evaluate performance for this epoch
            train_cost = sum([cost_function(self.predict(x), y) for x, y in training_set]) / len(training_set)
            valid_cost = sum([cost_function(self.predict(x), y) for x, y in validation_set]) / len(validation_set)
            train_accuracy = self.accuracy(training_set)
            valid_accuracy = self.accuracy(validation_set)
            print(f"epoch {epoch} complete! | train_cost: {np.around(train_cost, 5)} | train_accuracy: {np.around(train_accuracy, 5)} | valid_cost: {np.around(valid_cost, 5)} | valid_accuracy: {np.around(valid_accuracy, 5)}\n")
            
            if save_best_parameters:
                if valid_accuracy > best_valid_accuracy:
                    best_weights, best_biases = self.get_params()
                    self.set_best_params(best_weights, best_biases)
    
    def _back_prop(self, x, y, cost_function):
        nablaC_b = [np.zeros(layer.biases.shape) for layer in self.layers[1:]]
        nablaC_w = [np.zeros(layer.weights.shape) for layer in self.layers[1:]]
        delta = [np.zeros(layer.biases.shape) for layer in self.layers[1:]]
        output_activation_fn = self.get_layer(-1).fn
        
        # Forward pass
        weighted_inputs, activations = self._forward_prop(x)
        
        # Compute error for output layer
        if ((cost_function == cross_entropy) and (output_activation_fn == softmax)):
            delta[-1] = error_for_cross_entropy_softmax(activations[-1], y)
        else:
            delta[-1] = cost_derivative(cost_function, activations[-1], y) * \
            activation_derivative(self.get_layer(-1).fn, weighted_inputs[-1])        
        nablaC_b[-1] = delta[-1]
        nablaC_w[-1] = np.dot(delta[-1], activations[-2].transpose())  
        
        # Backpropagate error:
        for i in range(2, self.num_layers):
            delta[-i] = np.dot(self.get_layer(-i+1).weights.transpose(), delta[-i+1]) * \
            activation_derivative(self.get_layer(-i).fn, weighted_inputs[-i])
            nablaC_b[-i] = delta[-i]
            nablaC_w[-i] = np.dot(delta[-i], activations[-i-1].transpose())
            
        return nablaC_b, nablaC_w
        
    # Calculates the average gradients in one mini-batch, then updates the parameters
    # 1. Loop through the mini-batch, summing all the partial derivatives w.r.t all w, b 
    # 2. Divide the partial derivatives by the mini_batch_size and multiply by learning_rate
    # 3. decrement this value from the parameters
    def update_mini_batch(self, mini_batch, cost_function, learning_rate):
        sum_nablaC_b = [np.zeros(layer.biases.shape) for layer in self.layers[1:]]
        sum_nablaC_w = [np.zeros(layer.weights.shape) for layer in self.layers[1:]]
        
        for x,y in mini_batch:
            nablaC_b, nablaC_w = self._back_prop(x, y, cost_function)
            sum_nablaC_b = [snb + nb for snb, nb in zip(sum_nablaC_b, nablaC_b)]
            sum_nablaC_w = [snw + nw for snw, nw in zip(sum_nablaC_w, nablaC_w)]
            
        # Update step:
        # w = w - (learning_rate)*(sum_nabla_w / mini_batch_size)
        # b = b - (learning_rate)*(sum_nabla_b / mini_batch_size)
        for i in range(1, self.num_layers):
            self.get_layer(i).weights = self.get_layer(i).weights - learning_rate * (sum_nablaC_w[i-1] / len(mini_batch))
            self.get_layer(i).biases = self.get_layer(i).biases - learning_rate * (sum_nablaC_b[i-1] / len(mini_batch))
            
    def make_mini_batches(self, training_set, mini_batch_size):
        residual_batch_size = len(training_set) % mini_batch_size
        num_mini_batches = len(training_set) // mini_batch_size
        
        shuffled_training_set = np.copy(training_set)
        np.random.shuffle(shuffled_training_set)
        
        if (residual_batch_size > 0):
            mini_batches = np.split(shuffled_training_set[:-residual_batch_size], num_mini_batches)
            residual_batch = shuffled_training_set[-residual_batch_size:]
            mini_batches.append(residual_batch)
        else:
            mini_batches = np.split(shuffled_training_set, num_mini_batches)
        
        return mini_batches
    
    def accuracy(self, test_set):
        correct_predictions = 0
        for x, y in test_set:
            predicted_class = np.argmax(self.classify(x))
            actual_class = np.argmax(y)
            if actual_class == predicted_class:
                correct_predictions += 1
        return correct_predictions / len(test_set)