from setuptools import setup

setup(
    name='data-manager',
    version='0.0.0.dev0',
    description='Data manager for the evaluate-SOC-models project',
    author='Alexander S. Brunmayr',
    author_email='asb219@ic.ac.uk',
    url='https://github.com/asb219/evaluate-SOC-models/tree/main/data-manager',
    license='MIT license',
    packages=['data_manager', 'data_manager.file'],
    py_modules=[
        'data_manager.data',
        'data_manager.file.basefile',
        'data_manager.file.datafile',
        'data_manager.file.filefrom',
        'data_manager.file.utils'
    ]
)
