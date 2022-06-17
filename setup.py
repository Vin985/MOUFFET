import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="mouffet",
    author="Sylvain Christin",
    author_email="esc2203@umoncton.ca",
    description=(
        "A package offering a unified framework for training and evaluation machine"
        + " learning models"
    ),
    keywords="machine learning, unified framework, training, evaluation",
    # long_description=read('README.md'),
    version="1.0.0",
    packages=find_packages(),
    package_data={"": ["*.svg", "*.yaml", "*.zip", "*.ico", "*.bat"]},
    install_requires=["pandas", "feather-format", "pyyaml"],
)
