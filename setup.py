from setuptools import setup, find_packages

setup(
    name="ullu",
    version="0.1.0",
    description="Sanskrit Quote Retrieval Pipeline",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
)