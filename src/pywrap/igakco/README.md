# iGakco-SVM
## Installation

to use the python wrapper for the iGakco tool, make sure you are using python3 and then run:
```
pip install igakco
```

## Tutorial
 
Usage and method signature:
```
from igakco import igakco

igakco(g, m, trainfile, testfile, dictionary, labels, C=1, kernel_type=1, halt=None, kernelfile=None, modelfile=None, probability=False, threads=4, quiet=False, loadkernel=False, loadmodel=False)

    trainfile : filename for a set of training examples in FASTA format
    testfile : filename for a set of testing examples in FASTA format
    dictionary : file containing alphabet of characters that appear in the sequences (text file). If no file exists matching the specified file, the training dataset will be parsed and a dictionary by that filename will be created.
    labels : name of the file where the predicted labels for the test set will be placed (text file)

    g : gmer length; length of substrings (with up to m mismatches) used to compare sequences. Constraints: 0 < g < 20
    m : maximum number of mismatches to permit when comparing gmers. Constraints: 0 <= m < g
    threads : number of threads to use. Set to 1 to not parallelize kernel computation. 
    C : SVM C parameter. Default: 1.0
    probability : Flag for model to generate probability of class. Without it, AUC can't be calculated.
    quiet : If set, iGakco will run in quiet mode, only outputting specified files and only the accuracy values to stdout.
    kernelfile : Specify a kernel filename to print to. If loadkernel is also set, this will instead be used as the filename to load the kernel from
    modelfile : Specify a model filename to print to. If loadmodel is also set, this will instead be used as the filename to load the model from
    halt : set to 1 or 2. If 1, will halt the program after constructing and printing out the kernel. If 2, will halt after training and printing out the model
    loadkernel : If set, will load the train kernel from the file specified by -k
    loadmodel : If set, will load the train kernel from the file specified by -k and will load the model from the file specified by -o

```
Example:
Generates a kernel and prints it to kernel.txt, continues to train and predict as normal.

```
  igakco(7, 2, "trainingSet.fasta", "testSet.fasta", "protein.dictionary.txt", "labelsFile.txt", C=.01, probability=True, kernelfile="kernel.txt")
```
