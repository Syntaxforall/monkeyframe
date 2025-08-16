from setuptools import setup, find_packages
import os

# Safely read the README.md file, or use a default if it doesn't exist
try:
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A fast, minimal DataFrame library for ML workflows."

setup(
    name='monkeyframe',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20',
        'numba>=0.53',
    ],
    author='K.S.N.Ganesh',
    author_email='your.email@example.com',
    description='A fast, minimal DataFrame library for ML workflows.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Syntaxforall/monkeyframe',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
