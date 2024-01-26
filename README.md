# Matrix and Neural Network Library

This library provides a comprehensive implementation of matrices and basic neural network functionalities in C++. It includes the definition of a generic `Matrix` class, layer abstractions for neural networks, and various utility functions.

## Features

- **Matrix Operations**: Supports basic matrix operations such as addition, subtraction, multiplication, and transpose.
- **Neural Network Layers**: Implements layers like Linear (fully connected) and ReLU activation layers.
- **Backpropagation Support**: Includes functions for the backward pass to compute gradients.
- **Loss Function**: Mean Squared Error (MSE) loss and its gradient computation.
- **Utility Functions**: Additional functions like `argmax` for classification tasks and `get_accuracy` for evaluating model performance.

## Installation

To use this library, simply include the provided code in your C++ project. Ensure you have a C++11 or later compiler.

## Usage

### Matrix Class

- Construct matrices with specified dimensions.
- Initialize matrices with values using an initializer list.
- Perform operations like `*`, `+`, `-`, and transpose.

```cpp
Matrix<int> M(3, 2, {1, 2, 3, 4, 5, 6});
```

### Neural Network Layers

- Create layers like `Linear` and `ReLU`.
- Forward and backward methods for each layer.
- Optimize the weights using the `optimize` function in the `Linear` layer.

```cpp
Linear<double> linearLayer(in_features, out_features, n_samples, seed);
```

### Training a Model

- Define a network using the `Net` class.
- Perform forward and backward passes.
- Optimize the network using gradient descent.

```cpp
Net<double> net(in_features, hidden_dim, out_features, n_samples, seed);
// ... training loop ...
```

### Loss and Accuracy

- Compute loss using `MSELoss`.
- Calculate model accuracy using `get_accuracy`.

```cpp
double loss = MSELoss(y_true, y_pred);
double accuracy = get_accuracy(y_true, y_pred);
```

## Example

Refer to the `main` function in the provided code for an example of training a simple neural network on XOR data.

## License

This library is released under the [MIT License](https://opensource.org/licenses/MIT).

---

## Contributing

Contributions to improve the library are welcome. Please follow the standard pull request process.

