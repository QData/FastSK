import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FastSK",
    version="0.0.1",
    author="QData Lab at the University of Virginia",
    author_email="yq2h@virginia.edu",
    description="A library for classifying sequence inputs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QData/FastSK",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=open('requirements.txt').readlines(),
)
