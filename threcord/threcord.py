"""Pytorch tfrecord reader and writer utils.
"""

# -*- coding: utf-8 -*-
import os
import glob
import multiprocessing
import pickle
import numpy as np
import torch as th
from torch.utils.data import Dataset
import tensorflow as tf
import tqdm
import cv2


class RecordsMaker(object):
    """Transform a torch dataset into tfrecords files.
    """
    def __init__(self, th_dataset, save_dir, dataset_name,
                 datatype, workers=1, shards_num=1, **kwargs):
        """

        Args:
            th_dataset: a th.utils.data.Dataset instance
            save_dir: path to a directory where to save tfrecords files
            dataset_name: the name of the dataset, used in naming files
            shards_num: number of tfrecords files
            datatype: a dict where key is key of th_dataset output,
                      value is the data type. At present, only 4 types
                      are supported: img, array, string, int.
                      img, array, string are tested.
                      Here is an example:
                      ```python
                      type_info = {
                          'img': 'img',
                          's': 'array',
                          'shape': 'array',
                          'exp': 'array'
                      }
                      ```
            workers: multiprocessing workers

            accepted data additional info (kwargs):
                shape: required by img and array

                Example:
                    img={'shape':(256,256,3)},
                    s={'shape':(256,256,3)},
                    shape={'shape':(256,256,3)},
                    exp={'shape':(256,256,3)}
        """

        assert isinstance(th_dataset, Dataset), "Need a torch.util.data.Dataset"
        assert workers >= 0, "Number of workers should be a positive integer."
        assert shards_num >= 0, "Number of shards should be a positive integer."

        self.dataset_name = dataset_name
        self.save_dir = save_dir
        self.shards_num = shards_num
        self.length = len(th_dataset)

        self.shards_size = self.length // self.shards_num

        index_list = list(range(self.length))
        self.shard_list = [index_list[x: x+self.shards_size]
                           for x in range(0, self.length, self.shards_size)]
        self.th_dataset = th_dataset
        self.datatype = datatype
        self.workers = workers

        for i in self.datatype:
            if self.datatype[i] not in ['img', 'array', 'string', 'int']:
                raise TypeError
            if self.datatype[i] in ['img', 'array']:
                assert 'shape' in kwargs[i], "shape argument is required for img and array"

        with open(os.path.join(self.save_dir, 'dataset_config.pkl'), 'wb') as file:
            pickle.dump({
                'datatype': self.datatype,
                'info': kwargs
            }, file)

        if workers > 1:
            self.pool = multiprocessing.Pool(processes=workers)

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _array_feature(self, value):
        return self._float_feature(value.reshape(-1))

    def _img_feature(self, img):
        """

        Args:
            img: RGB Image, numpy array，ranging from 0 to 1

        Returns:
            th.train.Feature

        """
        img = (img * 255).astype(np.uint8)[:, :, ::-1]
        _, img_raw = cv2.imencode('.jpg', img)
        img_raw = img_raw.tostring()
        return self._bytes_feature(img_raw)

    def __len__(self):
        return len(self.th_dataset)

    def _process(self, idx):
        data = self.th_dataset.__getitem__(idx)
        feat = {}
        for key in data:
            if self.datatype[key] == 'img':
                feat.update({key: self._img_feature(data[key])})
            elif self.datatype[key] == 'array':
                feat.update({key: self._array_feature(data[key])})
            elif self.datatype[key] == 'string':
                feat.update({key: self._bytes_feature(data[key])})
            elif self.datatype[key] == 'int':
                feat.update({key: self._int64_feature(data[key])})
            else:
                raise NotImplementedError
        example = tf.train.Example(features=tf.train.Features(feature=feat))
        return example

    def run(self):
        """

        Returns:

        """
        if self.workers == 1:
            for shard_idx in range(self.shards_num):
                self._process_shards(shard_idx)

        elif self.workers > 1:
            for _ in tqdm.tqdm(
                    self.pool.imap_unordered(
                        self._process_shards, range(self.shards_num)),
                    total=self.shards_num):
                pass
            self.pool.close()
            self.pool.join()

        else:
            raise NotImplementedError

    def _process_shards(self, shard_idx):
        index_list = self.shard_list[shard_idx]
        file_name = '{}_{}.{}'.format(self.dataset_name, shard_idx, 'tfrecords')
        save_path = os.path.join(self.save_dir, file_name)
        with tf.python_io.TFRecordWriter(
            save_path, options=tf.python_io.TFRecordOptions(2)
        ) as writer:
            for idx in index_list:
                features = self._process(idx)
                writer.write(features.SerializeToString())

    def __getstate__(self):
        # pool instance cannot be pickled
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


class RecordsLoader(object):
    """TF record reader.
    """
    def __init__(self, batch_size, dataset_dir, epochs, parallel_calls,
                 shuffle_size, augmentation=None):
        """

        Args:
            batch_size:
            dataset_dir: e.g. /path/to/tfrecord/
            epochs:
            parallel_calls: number of workers
            shuffle_size: shuffle size for tensorflow reader
            augmentation:
        """
        with open(os.path.join(dataset_dir, 'dataset_config.pkl'), 'rb') as file:
            config = pickle.load(file)

        self.datatype = config['datatype']
        self.info = config['info']

        self.augmentation = augmentation

        self.batch_size = batch_size
        self.dataset_dir = glob.glob(os.path.join(dataset_dir, '*.tfrecords'))
        self.epochs = epochs
        self.parallel_calls = parallel_calls
        self.shuffle_size = shuffle_size

        # The usage of CUDA_VISIBLE_DEVICES here is to prevent tensorflow
        # session from occupying any GPU memory - we use it only for I/O
        # purposes
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
        ))
        del os.environ["CUDA_VISIBLE_DEVICES"]

        self.keys_to_features = {}
        for key, val in self.datatype.items():
            if val == 'img':
                item = {key: tf.FixedLenFeature(shape=(), dtype=tf.string)}
            elif val == 'array':
                item = {key: tf.FixedLenFeature(shape=self.info[key]['shape'], dtype=tf.float32)}
            elif val == 'string':
                item = {key: tf.FixedLenFeature(shape=(), dtype=tf.string)}
            elif val == 'int':
                item = {key: tf.FixedLenFeature(shape=(), dtype=tf.int32)}
            else:
                raise NotImplementedError
            self.keys_to_features.update(item)

        # Build I/O graph
        self.examples = self._build_graph()

    def _parser(self, record):
        """

        Args:
            record:

        Returns:

        """
        features = tf.parse_single_example(record, features=self.keys_to_features)

        feature_dict = {
            k: self._img_parser(img_raw=features[k], key=k) if v == 'img' else features[k]
            for k, v in self.datatype.items()
        }
        return feature_dict

    def _img_parser(self, img_raw, key):
        buffer = tf.cast(tf.image.decode_jpeg(img_raw, channels=3), tf.uint8)
        buffer = tf.reshape(buffer, shape=self.info[key]['shape'])
        return buffer

    def _build_graph(self):
        dataset = tf.data.TFRecordDataset(self.dataset_dir, compression_type='GZIP')
        dataset = dataset.map(
            lambda x: self._parser(x),
            num_parallel_calls=self.parallel_calls)
        dataset = dataset.shuffle(self.shuffle_size)
        dataset = dataset.repeat(
            count=self.epochs
        )
        dataset = dataset.batch(self.batch_size)

        iterator = dataset.make_one_shot_iterator()
        example = iterator.get_next()

        return example

    def __next__(self):
        try:
            example = self.sess.run(self.examples)
            if self.augmentation is not None:
                example = self.augmentation(example)
            for k, ary in example.items():
                example[k] = th.from_numpy(ary)
            return example
        except StopIteration:
            raise StopIteration

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()
