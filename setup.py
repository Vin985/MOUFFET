import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    author="Vin985",
    description=(
        "A library offering a unified framework for training and evaluation machine"
        + " learning models"
    ),
    keywords="machine learning, unified framework, training, evaluation",
    # long_description=read('README.md'),
    name="mouffet",
    version="0.9",
    packages=find_packages(),
    package_data={"": ["*.svg", "*.yaml", "*.zip", "*.ico", "*.bat"]},
)
