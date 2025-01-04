from setuptools import setup, find_packages

setup(
    name="brain_neuroimage_processing",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'pillow',
        'tqdm'
    ]
) 