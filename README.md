# Threcord, a pytorch utility for tfrecord

## Prerequisites
- OpenCV
- Tensorflow (1.8.0 Tested)
- Pytorch (0.3 Tested)
- tqdm (for progress bar)
- numpy

## Installation
```bash
git clone
cd threcord
pip install .
```

## Usage

- RecordsMaker

```python
from threcord import RecordsMaker

# create a torch dataset
# the dataset should return an example as a dict
# e.g.
# {
#     'img': img,
#     'label': label
# }

dataset = ExampleDataSet()

# prepare type information
# type info should be a dict share the same keys with dataset output
# only 4 accepted data type here: img, array, string, int
# you could implement any more of them.

datainfo = {
    'img': 'img',
    'label': 'array'
}

maker = RecordsMaker(
    th_dataset=dataset,
    save_dir='path/to/save/threcords/file',
    dataset_name='example_dataset',
    datatype=datainfo,
    workers=4,
    shards_num=4,
    # use kwargs to provide additional information need by the RecordsMaker
    # At present, shape argument is needed for img and array data type.
    img={'shape': (256,256,3)},
    label={'shape': (256,256,3)}
)

# run the maker and wait patient.
maker.run()
```

- RecordsLoader

```python
from threcord import RecordsLoader

# Create DataAug class if in need
# aug = DataAug(...)
# the augmentation class instance should accept a dict
# and return a dict with same keys while having new augmented values.

# create RecordsLoader instance first
loader = RecordsLoader(
    batch_size=32,
    dataset_dir='path/to/dataset/',
    epochs=10,
    parallel_calls=4,
    shuffle_size=200,
    # if there is data augmentation
    # augmentation = aug
    augmentation=None
)

for example in loader:
    # example is a dict whose content is th.Tensor
    # loader will raise StopIteration after the epochs designated.
    break
```