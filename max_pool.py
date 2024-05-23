import numpy as np

class MaxPool:

    def forward_pass(self, X, filter_size = 2, stride = 2):
        n, h, w, c = X.shape # number of examples, height, width, channel


        new_h = (h-filter_size) // stride +1
        new_w = (w-filter_size) // stride +1

        Z = np.zeros((n, new_h, new_w, c))

        for h in range(new_h):
            for w in range(new_w):
                    sliced_example = X[:, h*stride:h*stride+filter_size, w*stride:w*stride+filter_size, :]
                    Z[:, h, w, :] = np.max(sliced_example, axis=(1,2))


        cache = (X, filter_size, stride)

        return Z, cache

    def backprop(self, pooled, pooled_cache, dL_dx):
        grad_pool_out = dL_dx.reshape(pooled.shape)

        (X, filter_size, stride) = pooled_cache



        # Backprop through max pooling layer
        grad_conv_out = np.zeros_like(X)
        for i in range(pooled.shape[1]):
            for j in range(pooled.shape[2]):
                pool_region = X[:, i * stride :i * stride + filter_size,
                              j * stride :j * stride + filter_size, :]
                max_mask = (pool_region == pooled[:, i, j, :][:, None, None, :])
                grad_conv_out[:, i * stride :i * stride + filter_size,
                              j * stride :j * stride + filter_size, :] += max_mask * grad_pool_out[:, i, j, :][:, None, None, :]

        return grad_conv_out