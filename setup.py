import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neural_pipeline",
    version="0.0.1",
    author="Anton Fedotov",
    author_email="anton.fedotov.af@gmail.com.com",
    description="PyTorch pipeline for neural networks faster training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="none",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
