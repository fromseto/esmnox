from setuptools import setup, find_packages

setup(
    name="esmnox",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "equinox",
        "torch",  # for weight loading
    ],
    author="Lin Wang",
    description="ESM2 implementation in JAX/Equinox",
) 