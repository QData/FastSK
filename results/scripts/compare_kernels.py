#script for comparing kernels to a margin of error of .01
import matplotlib
matplotlib.use('Agg')
import fileinput
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import sys
import os

def compare_kernels(kernel1, kernel2, figure, tolerance=0.01):
    graph = []
    with open(kernel2, 'r') as file1:
        with open(kernel1, 'r') as file2:
            mismatch = 0
            total = 0
            line = 0
            while(True):
                linemm = []
                curr1 = file1.readline()
                curr2 = file2.readline()
                if not curr1 or not curr2:
                    break
                #split each corresponding line of the two files into an array of each float value (with index)
                arr1 = curr1.split(" ")
                arr2 = curr2.split(" ")
                for i in range(len(arr1)):
                    total+=1
                    colon = arr1[i].find(":")
                    if colon == -1:
                        continue
                    if abs(float(arr1[i][colon+1:]) - float(arr2[i][colon+1:])) > float(tolerance):
                        #print(arr1[i] + "  vs  " + arr2[i])
                        mismatch+=1
                        linemm.append(1)
                    else:
                        linemm.append(0)
                graph.append(linemm)
            print("\nTotal: " + repr(total) + "\nMismatch: " + repr(mismatch))

    data = np.array(graph)
    length = data.shape[0]
    width = data.shape[1]
    x, y = np.meshgrid(np.arange(length), np.arange(width))

    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1, projection='3d')
    #ax.plot_surface(x, y, data)
    plt.imshow(data, interpolation='nearest', cmap="Greys")
    plt.savefig(figure)

if __name__ == '__main__':
    if (len(sys.argv) != 4):
        print("\tUsage: python compare_kernels.py kernel1.txt kernel2.txt figure.png\n")
        print("\tkernel1.txt, kernel2.txt: the two kernels you wish to compare")
        print("\tfigure.png: name of the file to save a picture showing where the kernels differ")
        exit()
    kernel1 = sys.argv[1]
    kernel2 = sys.argv[2]
    figure = sys.argv[3]
    compare_kernels(kernel1, kernel2, figure)