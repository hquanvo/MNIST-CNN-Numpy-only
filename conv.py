import numpy as np

class Conv3x3:

    def __init__(self, num_filters, filter_size = 3):
        self.num_filters = num_filters
        # 8 x 3 x 3
        self.W = np.random.randn(num_filters, filter_size, filter_size) / 9
        self.b = np.zeros((num_filters, 1))

    def pad(self, X, pad_val = 1):
        # pad all images of dataset X
        padded_X = np.pad(X, ((0,0), (pad_val, pad_val), (pad_val, pad_val)), mode="constant")

        return padded_X

    def forward_pass(self, X, filter_size = 3, stride = 1, pad = 1):
        c, f, f = self.W.shape

        n, h, w = X.shape

        output_h = (h-f+ 2 * pad) // stride + 1
        output_w = (w - f + 2 * pad) // stride + 1

        Z = np.zeros((n, output_h, output_w, self.num_filters))

        X_pad = self.pad(X, pad)


        for i in range(n):
            for h in range(output_h):
                for w in range(output_w):
                    sliced_part = X_pad[i, h * stride:h * stride + filter_size, w * stride:w * stride + filter_size]
                    for k in range(c):
                        weights = self.W[k, :, :]

                        Z[i, h, w, k] = np.sum((sliced_part * weights)) + self.b[k]


        cache = (X, filter_size, stride, pad)
        return Z, cache

    def backprop(self, convolved, conv_cache, dL_dX_conv):
        dL_dw = np.zeros_like(self.W)
        dL_db = np.zeros_like(self.b)

        (X, filter_size, stride, pad) = conv_cache

        n, H, W = X.shape

        _, h_out, w_out, _ = dL_dX_conv.shape

        X_pad = self.pad(X, pad)

        for i in range(n):
            for h in range(0, H, stride):
                for w in range(0, W, stride):
                    for k in range(self.num_filters):
                        dL_dw[k, :, :] += X_pad[i, h:h+filter_size, w:w + filter_size] * dL_dX_conv[i, h, w, k]
                        dL_db[k] += dL_dX_conv[i, h, w, k]

        return dL_dw, dL_db

    def update_parameters(self, dL_dw, dL_db, learning_rate = 0.01):
        self.W = self.W - learning_rate * dL_dw
        self.b = self.b - learning_rate * dL_db

        # print("Conv params:")
        # print(self.W)
        # print(self.b)