from setuptools import setup, find_packages

setup(
    name="saliencyserieslab",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "tensorflow >= 2.16.2",
        "numpy >= 1.26.4",
        "mrseql >= 0.0.4",
        "sktime[all_extras] >= 0.34.0",
        "mlflow >= 2.17.0",
        "mlflow-skinny >= 2.17.0",
        "keras >= 3.6.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for time series analysis",
    keywords="saliency, series, analysis",
)
