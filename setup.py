from setuptools import setup, find_packages

# Read the contents of requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="bc_search_engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
)