import sys
import os

"""
If not a number, return false.
If it is a number, make sure it is either -1, 0, or 1.
"""


def isLabel(line, line_num):
    line = "".join(line.split())
    try:
        val = int(line)
        if not (val == -1 or val == 0 or val == 1):
            print(
                "Line number {} contained label of {}, however labels must be either -1 or 0 (negative class) or 1 (positive class).".format(
                    line_num, val
                )
            )
            exit()
        return True
    except ValueError:
        return False


"""
Read a fasta file in the following format:
	label (-1, 0, or 1)
	sequence (spanning any number of lines)
	etc
E.g.,
	1
	AAATG
	AAAAAAAG
	TTT
	0
	A
	AAAAA
Which results in:
	>1
	AAATGAAAAAAAGTTT
	>0
	AAAAAA
"""


def readFile(filename):
    data = []
    line_num = 1
    with open(filename) as f:
        # assumes the first line is a class label in {-1, 0, 1} (handle this line first)
        line = f.readline()
        if not line:
            print("Data file was empty")
            exit()
        line = "".join(line.split())
        label_line = ">" + line
        data.append(label_line)
        seq = ""
        line = f.readline()
        # loop through file. Accumulate sequences in the "seq" variable.
        # Sequence finishes when either end of file is reached or next label is rached
        while line:
            line = "".join(line.split())
            line_num += 1
            if isLabel(line, line_num):
                data.append(seq)
                label_line = ">" + line
                data.append(label_line)
                seq = ""
            else:
                seq += line
            line = f.readline()
        data.append(seq)
    return data


"""
Write data out to file in format accepted by fastsk or fastsk
"""


def writeFile(filename, data_array):
    with open(filename, "w+") as f:
        for line in data_array:
            f.write(line + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fastsk_formatter.py <file.fasta> <fastsk_format.fasta>")
        print(
            "\n<file.fasta> : fasta file in the following format: class label (-1, 0, or 1) followed by a sequence (which can span any number of lines)"
        )
        print("For example:\t\n\t1\n\tAAAA\n\tTTTTTTTT\n\t0\n\tAGT\n\tACC\n\tC")
        print(
            "<fastsk_format.fasta> : data from <file.fasta> converted to a form that can be read by fastsk or fastsk. If this file already exists, it will be overwritten."
        )
        exit()
    old_file = sys.argv[1]
    fastsk_file = sys.argv[2]
    fastsk_data = readFile(old_file)
    writeFile(fastsk_file, fastsk_data)
