from setuptools import setup

setup(
    name='threcord',
    version='0.1',
    description='Pytorch utils for tfrecord dataset.',
    author='pyc',
    install_requires=[
        'numpy',
        'tensorflow',
        'torch',
        'tqdm'
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    zip_safe=False
)