import os

from setuptools import find_packages, setup


def fetch_requirements(file):
    with open(file, 'r') as f:
        return [r.strip() for r in f.readlines()]


def fetch_readme():
    thisdir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(thisdir, 'README.md'), encoding='utf-8') as fin:
        readme_text = fin.read()
    return readme_text


setup(
    name='cubework',
    version='0.1.0',
    packages=find_packages(exclude=['benchmark']),
    description="Cubework library",
    long_description=fetch_readme(),
    long_description_content_type="text/markdown",
    license='Apache Software License 2.0',
    install_requires=fetch_requirements('requirements.txt'),
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Environment :: GPU :: NVIDIA CUDA',
    ],
)
