import fileinput

def main():
    sequences = "/mnt/c/Users/student/gakco/eamon/data/1.1.train.fasta"
    test = "/mnt/c/Users/student/gakco/eamon/data/1.1.test.fasta"
    svs = "/mnt/c/Users/student/gakco/eamon/fullmodel.txt"
    i=0
    output = open("/mnt/c/Users/student/gakco/eamon/SV.fasta", 'w+')
    with open(sequences, 'r') as fasta:
        with open(svs, 'r') as svs:
            fasta_ind = 0
            sv_count = 0
            curr_sv = 0
            #advance the svs pointer to the first line containing an sv index
            while(True):
                sv = svs.readline()
                if sv.find(":") != -1:
                    curr_sv = int(sv[sv.find(":") + 1 :])
                    break
            #now only copy over support vectors and training examples
            while(True):
                label = fasta.readline()
                string = fasta.readline()
                if not label:
                    break
                if curr_sv == fasta_ind:
                    #print(curr_sv)
                    output.write(label)
                    output.write(string)
                    sv = svs.readline()

                fasta_ind+=1
                if not sv:
                    break
                #print(sv)
                curr_sv = int(sv[sv.find(":") + 1 :])

    #now concatenate the test sequences
    output.close()
    output = open("/mnt/c/Users/student/gakco/eamon/SV.fasta", 'a')
    test_file = open(test, 'r')
    for line in test_file.readlines():
        output.write(line)
    output.close()
    test_file.close()



if __name__ == '__main__':
    main()