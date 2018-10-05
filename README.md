# iGakco-SVM
## Installation
Download and extract this repository, then enter:
```
  $ cd src
  $ make
```
An executable file named `iGakco` will now be located in the `iGakco-SVM/src` directory.

## Tutorial
iGakco-SVM takes several parameters:
        
        Usage: ./iGakco [options] <trainingFile> <testingFile> <dictionaryFile> <labelsFile>
          g : gmer length; length of substrings (with up to m mismatches) used to compare sequences. Constraints: 0 < g < 20
          m : maximum number of mismatches to permit when comparing gmers. Constraints: 0 <= m < g
          t : (default 4) number of threads to use. Set to 1 to not parallelize kernel computation. 
          C : (optional) SVM C parameter. Default: 1.0
          k : (optional) Specify a kernel filename to print to. If -l is also set, this will instead be used as the filename to load the kernel from
          o : (optional) Specify a model filename to print to. If -s is also set, this will instead be used as the filename to load the model from
          h : (optional) set to 1 or 2. If 1, will halt the program after constructing and printing out the kernel. If 2, will halt after training and printing out the model
          NO ARGUMENT FLAGS:
          l : (optional) If set, will load the train kernel from the file specified by -k
          s : (optional) If set, will load the train kernel from the file specified by -k and will load the model from the file specified by -o
          p : (optional) Flag for model to generate probability of class. Without it, AUC can't be calculated.
          q : (optional) If set, iGakco will run in quiet mode, only outputting specified files and only the accuracy values to stdout. 

          trainingFile : set of training examples in FASTA format
          testingFile : set of testing examples in FASTA format
          dictionaryFile : file containing alphabet of characters that appear in the sequences (text file). If no file exists matching the specified file, the training dataset will be parsed and a dictionary by that filename will be created.
          labelsFile : name of the file where the predicted labels for the test set will be placed (text file)
          kernelFile : where to write kernel matrix computed by iGakco (text file)


Examples:
Generates a kernel and prints it to kernel.txt, continues to train and predict as normal.
```
  $ ./iGakco -g 7 -m 2 -p -t 4 -C .01 -k kernel.txt trainingSet.fasta testSet.fasta protein.dictionary.txt labelsFile.txt
```

Loads the kernel from kernel.txt and trains a model with a new C value, outputting that model to model.txt, then predicts as normal
```
  $ ./iGakco -g 7 -m 2 -p -t 4 -C .1 -k kernel.txt -o model.txt -s trainingSet.fasta testSet.fasta protein.dictionary.txt labelsFile.txt
```
Loads the model from model.txt, and uses it to predict on the provided test set.
```
  $ ./iGakco -g 7 -m 2 -p -t 4 -C .1 -k kernel.txt -o model.txt -s trainingSet.fasta testSet.fasta protein.dictionary.txt labelsFile.txt
```

