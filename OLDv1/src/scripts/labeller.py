

def main():
    kernel1 = "/mnt/c/Users/student/gakco/eamon/kernel_test.txt"
    label = "/mnt/c/Users/student/gakco/eamon/test.labels.txt"
    #kernel1 ="/mnt/c/Users/student/gakco/eamon/kernel.txt"
    #kernel2 = "/mnt/c/Users/student/gakco/alibsvm/kernel.txt"
    out = open("TestkernelLabel.txt", 'w')
    with open(label, 'r') as labels:
        with open(kernel1, 'r') as kernel:
            i=1
            while(True):
                lab = labels.readline()
                k = kernel.readline()
                if not lab or not k:
                    break
                
                string = lab.strip() + " " + "0:" + repr(i) + " " + k.strip() + "\n"
                #string = lab.strip() + " " + k.strip() + "\n"
                out.write(string)
                i +=1

    out.close()


if __name__ == '__main__':
    main()