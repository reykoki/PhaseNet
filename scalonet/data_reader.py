import tensorflow as tf
import pywt
import scipy
import numpy as np

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import logging
import os

def get_scaleo(wfs, waveletname = 'mexh'):
    scales = np.array([2.5, 3, 4, 5, 6, 8, 10, 15, 20, 30])
    #scales = np.array([2.5, 4, 5, 10, 20, 30])
    all_powers = []
    for wf in wfs:
        all_powers.append(np.asarray(wf))
        [coefficients, freq] = pywt.cwt(data=wf,
                                        scales=scales,
                                        wavelet=waveletname,
                                        sampling_period=1/100)
        power = np.log2(abs(coefficients)+1e-10)
        if np.max(power) - np.min(power) > 0:
            norm_power= (power - np.min(power)) / (np.max(power) - np.min(power))
        else:
            norm_power = 0*power
        all_powers.extend(np.array(norm_power))
    X_shape = 3*(len(scales) + 1)
    return np.array(all_powers), X_shape

def py_func_decorator(output_types=None, output_shapes=None, name=None):
    def decorator(func):
        def call(*args, **kwargs):
            nonlocal output_shapes
            # flat_output_types = nest.flatten(output_types)
            flat_output_types = tf.nest.flatten(output_types)
            # flat_values = tf.py_func(
            flat_values = tf.numpy_function(func, inp=args, Tout=flat_output_types, name=name)
            if output_shapes is not None:
                for v, s in zip(flat_values, output_shapes):
                    v.set_shape(s)
            # return nest.pack_sequence_as(output_types, flat_values)
            return tf.nest.pack_sequence_as(output_types, flat_values)

        return call

    return decorator


def dataset_map(iterator, output_types, output_shapes=None, num_parallel_calls=None, name=None, shuffle=False):
    dataset = tf.data.Dataset.range(len(iterator))
    if shuffle:
        dataset = dataset.shuffle(len(iterator), reshuffle_each_iteration=True)

    @py_func_decorator(output_types, output_shapes, name=name)
    def index_to_entry(idx):
        return iterator[idx]

    return dataset.map(index_to_entry, num_parallel_calls=num_parallel_calls)


def normalize(data, axis=(0,)):
    """data shape: (nt, nsta, nch)"""
    data -= np.mean(data, axis=axis, keepdims=True)
    std_data = np.std(data, axis=axis, keepdims=True)
    std_data[std_data == 0] = 1
    data /= std_data
    # data /= (std_data + 1e-12)
    return data

class DataConfig:

    seed = 123
    use_seed = True
    n_channel = 33
    n_class = 3
    sampling_rate = 100
    dt = 1.0 / sampling_rate
    X_shape = [3001, 1, n_channel]
    Y_shape = [3001, 1, n_class]
    min_event_gap = 3 * sampling_rate
    label_shape = "gaussian"
    label_width = 30
    dtype = "float32"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DataReader:
    def __init__(self, data_dict, format="numpy", config=DataConfig(), **kwargs):

        self.dtype = config.dtype
        self.X_shape = config.X_shape
        self.Y_shape = config.Y_shape
        self.data = data_dict
        self.num_data = len(self.data['X'])
        self.config = config

    def __len__(self):
        return self.num_data

    def get_data_shape(self):
        return self.data['X'][0].shape

    def __getitem__(self, i):
        wfs = self.data['X'][i]
        labels = self.data['y'][i]
        scalo, X_shape = get_scaleo(wfs)
        scalo = np.reshape(np.transpose(scalo), self.X_shape)
        labels = np.reshape(labels, self.Y_shape)
        return (scalo.astype(self.dtype), labels.astype(self.dtype), "basename")

    def dataset(self, batch_size, num_parallel_calls=2, shuffle=True, drop_remainder=True):
        dataset = dataset_map(
            self,
            output_types=(self.dtype, self.dtype, "string"),
            output_shapes=(self.X_shape, self.Y_shape, None),
            num_parallel_calls=num_parallel_calls,
            shuffle=shuffle,
        )
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(batch_size * 2)
        return dataset


