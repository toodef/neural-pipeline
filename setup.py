import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neural_pipeline",
    version="0.0.1",
    author="Anton Fedotov: @toodef",
    author_email="anton.fedotov.af@gmail.com.com",
    description="Neural Networks train pipeline based on PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="none",
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=['numpy', 'tensorboardX', 'tqdm', 'torch==0.4.1'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        "Operating System :: OS Independent",
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
