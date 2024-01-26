#include <cmath>
#include <initializer_list>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include <algorithm>

template <typename T>
class Matrix {
private:
    std::vector<T> data;
    int rows, cols;

public:
    // Default constructor
    Matrix() : rows(0), cols(0) {}

    // Constructor with dimensions
    Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows * cols, T()) {}

    // Constructor with initializer list
    Matrix(int rows, int cols, const std::initializer_list<T>& list) : rows(rows), cols(cols), data(list) {
        if (list.size() != rows * cols) {
            throw std::invalid_argument("Initializer list size does not match matrix dimensions.");
        }
    }

    // Copy constructor
    Matrix(const Matrix& other) : data(other.data), rows(other.rows), cols(other.cols) {}

    // Move constructor
    Matrix(Matrix&& other) noexcept : data(std::move(other.data)), rows(other.rows), cols(other.cols) {
        other.rows = 0;
        other.cols = 0;
    }

    // Copy assignment operator
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            data = other.data;
            rows = other.rows;
            cols = other.cols;
        }
        return *this;
    }

    // Move assignment operator
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            rows = other.rows;
            cols = other.cols;

            other.rows = 0;
            other.cols = 0;
        }
        return *this;
    }

    // Destructor
    ~Matrix() {}

    // Access operator
    T& operator[](const std::pair<int, int>& ij) {
        if (ij.first >= rows || ij.second >= cols || ij.first < 0 || ij.second < 0) {
            throw std::out_of_range("Matrix index out of range.");
        }
        return data[ij.first * cols + ij.second];
    }

    // Constant access operator
    const T& operator[](const std::pair<int, int>& ij) const {
        if (ij.first >= rows || ij.second >= cols || ij.first < 0 || ij.second < 0) {
            throw std::out_of_range("Matrix index out of range.");
        }
        return data[ij.first * cols + ij.second];
    }

    // Matrix-scalar multiplication
    template<typename U>
    Matrix<typename std::common_type<T, U>::type> operator*(U x) const {
        Matrix<typename std::common_type<T, U>::type> result(rows, cols);
        for (int i = 0; i < rows * cols; ++i) {
            result.data[i] = data[i] * x;
        }
        return result;
    }

    // Matrix-matrix multiplication
    template<typename U>
    Matrix<typename std::common_type<T, U>::type> operator*(const Matrix<U>& B) const {
        if (cols != B.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        Matrix<typename std::common_type<T, U>::type> result(rows, B.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < B.cols; ++j) {
                for (int k = 0; k < cols; ++k) {
                    result.data[i * B.cols + j] += (*this)[{i, k}] * B[{k, j}];
                }
            }
        }
        return result;
    }

    // Matrix addition
    template<typename U>
    Matrix<typename std::common_type<T, U>::type> operator+(const Matrix<U>& B) const {
        if (rows != B.rows || cols != B.cols) {
            if (!(rows == 1 || B.rows == 1) || cols != B.cols) {
                throw std::invalid_argument("Matrix dimensions do not match for addition.");
            }
        }

        int maxRows = std::max(rows, B.rows);
        Matrix<typename std::common_type<T, U>::type> result(maxRows, cols);

        for (int i = 0; i < maxRows; ++i) {
            for (int j = 0; j < cols; ++j) {
                T val1 = (i < rows) ? (*this)[{i, j}] : T();
                U val2 = (i < B.rows) ? B[{i, j}] : U();
                result[{i, j}] = val1 + val2;
            }
        }

        return result;
    }

    // Matrix subtraction
    template<typename U>
    Matrix<typename std::common_type<T, U>::type> operator-(const Matrix<U>& B) const {
        // Implementation similar to operator+ with subtraction logic
        if (rows != B.rows || cols != B.cols) {
            if (!(rows == 1 || B.rows == 1) || cols != B.cols) {
                throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
            }
        }

        int maxRows = std::max(rows, B.rows);
        Matrix<typename std::common_type<T, U>::type> result(maxRows, cols);

        for (int i = 0; i < maxRows; ++i) {
            for (int j = 0; j < cols; ++j) {
                T val1 = (i < rows) ? (*this)[{i, j}] : T();
                U val2 = (i < B.rows) ? B[{i, j}] : U();
                result[{i, j}] = val1 - val2;
            }
        }

        return result;

    }

    // Transpose
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result[{j, i}] = (*this)[{i, j}];
            }
        }
        return result;
    }

    // Getters for rows and columns
    int getRows() const { return rows; }
    int getCols() const { return cols; }

    // Print function
    void print_Matrix() const
    {
        for (int i = 0; i < rows; i++)
        {
            std::cout << "[";
            for (int j = 0; j < cols; j++)
            {
                std::cout << (*this)[{i, j}];
                if (j != cols - 1)
                {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;
        }
    }

};

// Abstract Layer class
template<typename T>
class Layer {
public:
    virtual ~Layer() {}
    virtual Matrix<T> forward(const Matrix<T>& x) = 0;
    virtual Matrix<T> backward(const Matrix<T>& dy) = 0;
};

// Linear Layer class
template<typename T>
class Linear : public Layer<T> {
private:
    Matrix<T> weights, biases, weightsGradient, biasesGradient, cache;
    int inFeatures, outFeatures, nSamples;

public:
    // Constructor
    Linear(int in_features, int out_features, int n_samples, int seed) 
        : inFeatures(in_features), outFeatures(out_features), nSamples(n_samples) {

        // Initialize the random number generators
        std::default_random_engine generator(seed);
        std::normal_distribution<T> distribution_normal(0.0, 1.0);
        std::uniform_real_distribution<T> distribution_uniform(0.0, 1.0);

        // Initialize matrices
        weights = Matrix<T>(in_features, out_features);
        biases = Matrix<T>(1, out_features);
        weightsGradient = Matrix<T>(in_features, out_features);
        biasesGradient = Matrix<T>(1, out_features);
        cache = Matrix<T>(n_samples, in_features);

        // Randomly initialize weights and biases
        for (int i = 0; i < in_features; ++i) {
            for (int j = 0; j < out_features; ++j) {
                weights[{i, j}] = distribution_normal(generator);
            }
        }
        for (int j = 0; j < out_features; ++j) {
            biases[{0, j}] = distribution_uniform(generator);
        }
    }

    // Destructor
    virtual ~Linear() {}

    // Forward function
    virtual Matrix<T> forward(const Matrix<T>& x) override final {
        cache = x;  // Store the input for use in backward pass
        return (x * weights) + biases;  // y = x * w + b
    }

    // Backward function
    virtual Matrix<T> backward(const Matrix<T>& dy) override final {
        // Compute gradients
        weightsGradient = cache.transpose() * dy;
        biasesGradient = dy;  // Sum of dy for each sample

        // Compute and return downstream gradient
        return dy * weights.transpose();
    }

    // Optimize function
    void optimize(T learning_rate) {
        weights = weights - (weightsGradient * learning_rate);
        biases = biases - (biasesGradient * learning_rate);
    }
};


// ReLU Layer class
template<typename T>
class ReLU : public Layer<T> {
private:
    Matrix<T> cache;
    int inFeatures, outFeatures, nSamples;

public:
    // Constructor
    ReLU(int in_features, int out_features, int n_samples) 
        : inFeatures(in_features), outFeatures(out_features), nSamples(n_samples) {
        // Initialize cache
        cache = Matrix<T>(n_samples, in_features);
    }

    // Destructor
    virtual ~ReLU() {}

    // Forward function
    virtual Matrix<T> forward(const Matrix<T>& x) override final {
        cache = x;  // Store the input for use in backward pass
        Matrix<T> result = x;

        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < inFeatures; ++j) {
                result[{i, j}] = std::max(T(0), x[{i, j}]);  // ReLU(x) = max(0, x)
            }
        }
        return result;
    }

    // Backward function
    virtual Matrix<T> backward(const Matrix<T>& dy) override final {
        Matrix<T> dx = dy;

        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < inFeatures; ++j) {
                dx[{i, j}] = cache[{i, j}] > T(0) ? dy[{i, j}] : T(0);  // dReLU(x)/dx = 1 if x > 0, else 0
            }
        }

        return dx;
    }
};

template <typename T>
class Net {
private:
    std::vector<std::shared_ptr<Layer<T>>> layers;

public:
    // Constructor
    Net(int in_features, int hidden_dim, int out_features, int n_samples, int seed) {
        // Create and add layers to the network
        layers.push_back(std::make_shared<Linear<T>>(in_features, hidden_dim, n_samples, seed));
        layers.push_back(std::make_shared<ReLU<T>>(hidden_dim, hidden_dim, n_samples));
        layers.push_back(std::make_shared<Linear<T>>(hidden_dim, out_features, n_samples, seed));
    }

    // Destructor
    ~Net() {}

    // Forward function
    Matrix<T> forward(const Matrix<T>& x) {
        Matrix<T> output = x;
        for (auto& layer : layers) {
            output = layer->forward(output);
        }
        return output;
    }

    // Backward function
    Matrix<T> backward(const Matrix<T>& dy) {
        Matrix<T> output = dy;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            output = (*it)->backward(output);
        }
        return output;
    }

    // Optimize function
    void optimize(T learning_rate) {
        for (auto& layer : layers) {
            // Dynamically cast to check if the layer is Linear (trainable)
            auto linearLayer = std::dynamic_pointer_cast<Linear<T>>(layer);
            if (linearLayer) {
                linearLayer->optimize(learning_rate);
            }
        }
    }
};


template <typename T>
T MSELoss(const Matrix<T>& y_true, const Matrix<T>& y_pred) {
    T sum = 0;
    int n = y_true.getRows() * y_true.getCols();
    for (int i = 0; i < y_true.getRows(); ++i) {
        for (int j = 0; j < y_true.getCols(); ++j) {
            T diff = y_pred[{i, j}] - y_true[{i, j}];
            sum += diff * diff;
        }
    }
    return sum / n;
}

template <typename T>
Matrix<T> MSEgrad(const Matrix<T>& y_true, const Matrix<T>& y_pred) {
    int n = y_true.getRows() * y_true.getCols();
    Matrix<T> grad(y_true.getRows(), y_true.getCols());
    for (int i = 0; i < y_true.getRows(); ++i) {
        for (int j = 0; j < y_true.getCols(); ++j) {
            grad[{i, j}] = 2 * (y_pred[{i, j}] - y_true[{i, j}]) / n;
        }
    }
    return grad;
}


// Calculate the argmax 
template <typename T>
Matrix<int> argmax(const Matrix<T>& y) {
    Matrix<int> indices(1, y.getRows()); // One row, number of columns equals number of rows in y
    for (int i = 0; i < y.getRows(); ++i) {
        int maxIndex = 0;
        T maxValue = y[{i, 0}];
        for (int j = 1; j < y.getCols(); ++j) {
            if (y[{i, j}] > maxValue) {
                maxValue = y[{i, j}];
                maxIndex = j;
            }
        }
        indices[{0, i}] = maxIndex;
    }
    return indices;
}

template <typename T>
T get_accuracy(const Matrix<T>& y_true, const Matrix<T>& y_pred) {
    Matrix<int> trueIndices = argmax(y_true);
    Matrix<int> predIndices = argmax(y_pred);

    int correct = 0;
    for (int i = 0; i < trueIndices.getCols(); ++i) {
        if (trueIndices[{0, i}] == predIndices[{0, i}]) {
            correct++;
        }
    }

    return static_cast<T>(correct) / trueIndices.getCols();
}


int main(int argc, char* argv[]) {
    
    // Set parameters
    double learning_rate = 0.005;
    int optimizer_steps = 100;
    int seed = 1;

    // Initializing XOR data
    Matrix<double> x_xor(4, 2, {0, 0, 0, 1, 1, 0, 1, 1});
    Matrix<double> y_xor(4, 2, {1, 0, 0, 1, 0, 1, 1, 0});

    // Network dimensions
    int in_features = 2, hidden_dim = 100, out_features = 2;

    // Initialize network
    Net<double> net(in_features, hidden_dim, out_features, 4, seed);

    // Training loop
    for (int i = 0; i < optimizer_steps; ++i) {
        // Forward pass
        Matrix<double> y_pred = net.forward(x_xor);

        // Compute loss and gradients
        double loss = MSELoss(y_xor, y_pred);
        Matrix<double> grad = MSEgrad(y_xor, y_pred);

        // Backward pass
        net.backward(grad);

        // Optimizer step
        net.optimize(learning_rate);

        // Calculate and store accuracy
        double accuracy = get_accuracy(y_xor, y_pred);
        std::cout << "Step " << i << ", Loss: " << loss << ", Accuracy: " << accuracy << std::endl;
    }

    return 0;
}