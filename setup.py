from setuptools import setup, find_packages

setup(
    name="anfispy",
    version="0.6.0",
    author="Matheus Zaia Monteiro",
    author_email="matheus.z.monteiro@gmail.com",
    url="https://github.com/mZaiam/ANFISpy",
    packages=find_packages(),
    install_requires=[
        "torch>=2.5.1",
        "numpy>=2.1.3",
        "matplotlib>=3.10.0",
        ],
    python_requires=">=3.10.8"
)
