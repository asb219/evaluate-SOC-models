from setuptools import setup

setup(
    name='evaluate-SOC-models',
    version='1.0.0',
    description='Evaluate new-gen SOC models with 14C',
    long_description='Evaluate the performance of new-generation soil organic'\
        ' carbon models with radiocarbon (14C) data of soil density fractions',
    author='Alexander S. Brunmayr',
    author_email='asb219@ic.ac.uk',
    url='https://github.com/asb219/evaluate-SOC-models',
    license='GNU General Public License v3 (GPLv3)',
    packages=[
        'evaluate_SOC_models',
        'evaluate_SOC_models.data',
        'evaluate_SOC_models.data.sources',
        'evaluate_SOC_models.models',
        'evaluate_SOC_models.plots'
    ],
    entry_points={'console_scripts': [
        'config = evaluate_SOC_models.config:main'
    ]},
    package_data={
        'evaluate_SOC_models.models': ['somic.r'],
        'evaluate_SOC_models': ['config_defaults.ini']
    }
)
