import os
import sys 

import numpy as np

class KFold:
    def __init__(self, root_path, include_n, k, test_size):
        self.i = 0
        self.k = k
        self.include_n = include_n
        self.test_size = test_size

        self.items = {}
        self.size = 0

        for entry in os.listdir(root_path):
            p = os.path.join(root_path, entry)
            if not os.path.isdir(p):
                sys.exit("expected class dir in root directory but got file")

            self.items[entry] = []
            for fname in os.listdir(p):
                fpath = os.path.join(p, fname)
                self.items[entry].append(fpath)
                self.size += 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.k: 
            raise StopIteration()

        start = self.i * self.include_n
        end = start + self.include_n

        x_train = []
        y_train = []
        x_test = []
        y_test = []

        for k, v in self.items.items():
            if end >= len(v): 
                raise StopIteration()

            x_train.extend(v[start:end])
            y_train.extend([k] * self.include_n)

            v_test = v[:start]
            v_test.extend(v[end:])

            if len(v_test) > self.test_size:
                v_test = v_test[:self.test_size]

            x_test.extend(v_test)
            y_test.extend([k] * len(v_test))

        # train_size = self.include_n * len(self.items.keys())
        # test_size = self.size - train_size

        # assert(len(x_train) == train_size)
        # assert(len(y_train) == train_size)

        # assert(len(x_test) == test_size)
        # assert(len(y_test) == test_size)

        train = Embeddings(x_train, y_train)
        test = Embeddings(x_test, y_test)

        self.i += 1
        return train, test

class Embeddings:
    def __init__(self, paths, y):
        super(Embeddings, self).__init__()
        self.x = []

        for path in paths:
            x = np.load(path).reshape(1, -1)[0]
            d = np.sqrt(np.sum(x ** 2))
            self.x.append(x / d)

        self.x = np.array(self.x)
        self.y = np.array(y)
