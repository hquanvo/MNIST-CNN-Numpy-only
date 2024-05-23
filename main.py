import copy
import time
from os.path import join
import numpy as np
from conv import Conv3x3
from loss import SoftmaxLoss
from max_pool import MaxPool
from mnist_data_loader import MnistDataloader

#
# Set file paths based on added MNIST Datasets
#
input_path = './input'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')


def train(conv, pool, softmax, X_train, y_train):
    # forward propagation step
    convolved, conv_cache = conv.forward_pass(X_train)

    pooled, pool_cache = pool.forward_pass(convolved)

    probs, softmax_cache = softmax.forward_pass(pooled)

    loss, accuracy = softmax.compute_cross_entropy_loss(probs, y_train)

    # back propagation step
    dL_dw_fc, dL_db_fc, dL_dX_pooled = softmax.backprop(probs, y_train, softmax_cache)

    dL_dX_conv = pool.backprop(pooled, pool_cache, dL_dX_pooled)

    dL_dw_conv, dL_db_conv = conv.backprop(convolved, conv_cache, dL_dX_conv)

    update_parameters(conv, softmax, dL_dw_conv, dL_db_conv, dL_dw_fc, dL_db_fc)

    return loss, accuracy


def test(conv, pool, softmax, X_test, y_test):
    conv_out, _ = conv.forward_pass(X_test)
    pool_out, _ = pool.forward_pass(conv_out)
    probs, _ = softmax.forward_pass(pool_out)

    loss, accuracy = softmax.compute_cross_entropy_loss(probs, y_test)

    return loss, accuracy


def update_parameters(conv, softmax, dW_conv, db_conv, dW_fc, db_fc, learning_rate=5e-2):
    conv.update_parameters(dW_conv, db_conv, learning_rate)
    softmax.update_parameters(dW_fc, db_fc, learning_rate)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #
    # Load MNIST dataset
    #
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    start = time.time()

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_train_set = x_train[:1000]
    y_train_set = y_train[:1000]

    permutation = np.random.permutation(len(y_train_set))
    x_train_set = x_train_set[permutation]
    y_train_set = y_train_set[permutation]

    x_validation_set = x_train[4000:4250]
    y_validation_set = y_train[4000:4250]

    conv = Conv3x3(8)
    pool = MaxPool()
    softmax = SoftmaxLoss(14 * 14 * 8, 10)

    min_loss = np.inf
    best_acc = 0

    best_conv = conv
    best_pool = pool
    best_softmax = softmax

    for i in range(15):
        print("Training epoch ", i + 1)

        batch_size = 250

        for j in range(int(1000/batch_size)):
            x_train_batch = x_train_set[j:(j+batch_size)]
            y_train_batch = y_train_set[j:(j+batch_size)]
            loss, accuracy = train(conv, pool, softmax, (x_train_batch / 255 - 0.5), y_train_batch)

        print("Accuracy: ", accuracy)
        print("Loss: ", loss)

        if (loss < min_loss):
            min_loss = loss
            best_acc = accuracy
            best_conv = copy.deepcopy(conv)
            best_pool = copy.deepcopy(pool)
            best_softmax = copy.deepcopy(softmax)

    final_loss, final_accuracy = test(best_conv, best_pool, best_softmax, (x_validation_set / 255 - 0.5),
                                      y_validation_set)

    print("Loss on validation set: ", final_loss)
    print("Accuracy on validation set: ", final_accuracy)

    end = time.time()

    print("Time elasped: ", end - start, " seconds")
