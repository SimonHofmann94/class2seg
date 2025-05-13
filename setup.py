from setuptools import setup, find_packages

setup(
    name="class2seg",
    version="0.1",
    packages=find_packages(),        # findet den Ordner `project/`
    install_requires=[
        "torch",
        "torchvision",
        "hydra-core",
        "tensorboard",
        "numpy",
        "omegaconf",
        "Pillow",
    ],
    # optional: entry_points für CLI
)