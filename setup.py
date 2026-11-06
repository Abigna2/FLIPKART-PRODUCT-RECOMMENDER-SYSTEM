
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="flipkart-product-recommender",
    version="1.0.0",
    author="Abigna",
    packages=find_packages(),
    install_requires=requirements,
)
