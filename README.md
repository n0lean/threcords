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
 




```