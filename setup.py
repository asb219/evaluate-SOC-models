from distutils.core import setup

setup(
    name='evaluate-SOC-models',
    version='0.1.0',
    description='Evaluate new-gen SOC models with 14C',
    long_description='Evaluate the performance of new-generation soil organic'\
        ' carbon models with radiocarbon (14C) data of soil density fractions',
    author='Alexander S. Brunmayr',
    author_email='asb219@ic.ac.uk',
    url='https://github.com/asb219/evaluate-SOC-models',
    license='GNU General Public License v3 (GPLv3)',
    packages=['evaluate_SOC_models'],
    modules=['config']
)
