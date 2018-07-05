#script for comparing kernels to a margin of error of .01
import matplotlib
matplotlib.use('Agg')
import fileinput
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np

def main():
    #kernel1 = "/home/eamon/repos/gakco/release/src/kernel.txt"
    kernel2 = "/home/eamon/repos/igakco/src/kernel.txt"
    kernel1 ="/home/eamon/repos/igakco/src/train_kernel.txt"
    #kernel2 = "/mnt/c/Users/student/gakco/alibsvm/kernel.txt"
    
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
                    if abs(float(arr1[i][colon+1:]) - float(arr2[i][colon+1:])) > float(0.000001):
                        #print(arr1[i] + "  vs  " + arr2[i])
                        mismatch+=1
                        if(len(graph)>1000 and len(graph) < 1010):
                            print(repr(len(graph)+1)+" " + arr1[i])
                        linemm.append(1)
                    else:
                        linemm.append(0)
                graph.append(linemm)
            print("\nTotal: " + repr(total) + "\nMismatch: " + repr(mismatch))

    max_len = len(graph[-1])
    data = np.array(graph)
    data = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in graph])
    length = data.shape[0]
    width = data.shape[1]
    x, y = np.meshgrid(np.arange(length), np.arange(width))

    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1, projection='3d')
    #ax.plot_surface(x, y, data)
    plt.imshow(data,
           interpolation='nearest', cmap="Greys")
    plt.savefig("/home/eamon/repos/igakco/src/compare.png")


if __name__ == '__main__':
    main()