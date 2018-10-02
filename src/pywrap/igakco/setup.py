import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="igakco",
    version="0.0.1",
    author="Ritambhara Singh, Eamon Collins",
    author_email="ec3bd@virginia.edu",
    description="A python wrapper for iGakco",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QData/iGakco-SVM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)