# iGakco-SVM
## Installation
Download and extract this repository, then enter:
```
  $ cd src
  $ cd make
```
An executable file named `GaKCo` will now be located in the `iGakco-SVM/src` directory.
## Tutorial
iGakco-SVM takes several parameters:
        
        Usage: ./GaKCo [options] <trainingFile> <testingFile> <dictionaryFile> <labelsFile> <kernelFile>
          g : gmer length; length of substrings (with up to m mismatches) used to compare sequences. Constraints: 0 < g < 20
          k : kmer length; length of non-gapped substrings within gmers. Constraints: k < g
          n : (optional) maxmimum number of examples in the dataset. Default: 15000
          t : (optional) number of threads to use. Set to 1 to not parallelize kernel computation. 
          C : (optional) SVM C parameter. Default: 1.0
          trainingFile : set of training examples in FASTA format
          testingFile : set of testing examples in FASTA format
          dictionaryFile : file containing alphabet of characters that appear in the sequences (text file)
          labelsFile : name of the file to place labels from the examples (text file). Used by iGakco to create kernel matrix
          kernelFile : where to write kernel matrix computed by iGakco (text file)
For example:
```
  $ ./GaKCo -g 7 -k 5 -n 20000 -t 4 -C 1.1 trainingSet.fasta testSet.fasta protein.dictionary.txt labelsFile.txt kernel.txt
```
