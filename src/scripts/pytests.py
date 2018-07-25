import os
import subprocess
import time
import datetime
import sys


igakco = False
gakco = False
gkm = True

datapath = '../data'
resultspath = '/localtmp/ec3bd/testresults'

datasets = [
			{'name':'1.1', 'g':7, 'm':2, 'c': .01},
			{'name':'1.34', 'g':10, 'm':9, 'c': .1},
			{'name':'2.19', 'g':7, 'm':6, 'c': 100},
			{'name':'2.31', 'g':10, 'm':3, 'c': 10},
			{'name':'2.1', 'g':10, 'm':7, 'c': 10},
			{'name':'2.34', 'g':7, 'm':1, 'c': .01},
			{'name':'2.41', 'g':10, 'm':4, 'c': .01},
			{'name':'2.8', 'g':10, 'm':9, 'c': 10},
			{'name':'3.19', 'g':8, 'm':7, 'c': .1},
			{'name':'3.25', 'g':10, 'm':2, 'c': 1},
			{'name':'3.33', 'g':10, 'm':5, 'c': .01},
			{'name':'3.50', 'g':10, 'm':3, 'c': .01},
			# {'name':'CTCF', 'g':10, 'm':5, 'c': 1},
			# {'name':'EP300', 'g':10, 'm':5, 'c': 1},
			# {'name':'JUND', 'g':10, 'm':3, 'c': 1},  
			# {'name':'RAD21', 'g':10, 'm':5, 'c': 1},
			# {'name':'SIN3A', 'g':10, 'm':3, 'c': 1},
		]




def recordTime(filepath, time):
	with open(filepath, "a+") as timefile:
		timefile.seek(0)
		lines = timefile.readlines()
		lines.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\t\t" + repr(time) +"\n")

		times = []
		for line in lines:
			if line.find(":") != -1 and line.strip() != "":
				times.append(float(line[line.find(":")+3:-1].strip()))
		avg = sum(times) / float(len(times))

		if lines[0].find("Average") == -1:
			lines.insert(0,"Average (s) \t" + repr(avg)+"\n")
		else:
			lines[0] = "Average (s): \t" +repr(avg) + "\n"

	with open(filepath, "w") as timefile:
		for line in lines:
			timefile.write(line)

def gkmify(sequencespath, posfile, negfile, test=False):
	uniqueID = 0
	posData = []
	negData = []
	with open(sequencespath) as f:
		line = f.readline()
		islabel = True
		pos = True
		while line:
			line = ''.join(line.split())
			if (islabel):
				label = ">"
				label += str(uniqueID)
				uniqueID += 1
				islabel = False
				pos = True if(line[1] == "1") else False
				if(pos):
					posData.append(label)
				else:
					negData.append(label)
			else:
				if(pos):
					posData.append(line)
				else:
					negData.append(line)
				islabel = True
			line = f.readline()

	if test:
		with open(posfile, 'w+') as f:
			for line in posData:
				f.write(line + "\n")
			for line in negData:
				f.write(line+"\n")
	else:
		with open(posfile, 'w+') as f:
			n = len(posData)
			for i in range(0, n):
				line = posData[i]
				f.write(line + "\n")
		with open(negfile, 'w+') as f:
			n = len(negData)
			for i in range(0, n):
				line = negData[i]
				f.write(line + "\n")


def test_igakco():
	#overwrite aggregated time doc so this runs results are only ones
	f = open(os.path.join(resultspath, 'igakco', 'aggtime.txt'), 'w+')
	f.close()

	for data in datasets:
		outputpath = os.path.join(resultspath, 'igakco', data['name'].replace(".", "_"))
		trainfile = os.path.join(datapath, data['name'] + ".train.fasta")
		testfile = os.path.join(datapath, data['name'] + ".test.fasta")
		if data['name'] == 'EP300' or data['name'] == 'CTCF'or data['name'] == 'JUND'or data['name'] == 'RAD21' or data['name'] == 'SIN3A':
			dictfile = os.path.join(datapath, "dna.dictionary.txt")
		else:
			dictfile = os.path.join(datapath, "protein.dictionary.txt")


		print()

		with open('sequences.fasta', 'w') as outfile:
			with open(trainfile) as infile:
				outfile.write(infile.read())
			with open(testfile) as infile:
				outfile.write(infile.read())

		#command = ["./iGakco", "-g", repr(data['g']), "-m", repr(data['m']), "-t", repr(20), '-C', repr(data['c']), "-p", "-k", os.path.join(outputpath, "kernel.txt"), '-o', os.path.join(outputpath, "model.txt"), 'sequences.fasta', 'sequences.fasta', dictfile, os.path.join(outputpath, "labels.txt")]
		command = "valgrind --tool=massif ./iGakco -h 1 -r 2 -g "+ repr(data['g']) + " -m " + repr(data['m']) + " -t "+ repr(20) + ' -C ' + repr(data['c']) + " -p " + " -k "+ os.path.join(outputpath, "kernel.txt") + ' -o ' + os.path.join(outputpath, "model.txt") +" "+ os.path.join(datapath, data['name']+".train.fasta")+ " "+os.path.join(datapath, data['name']+".test.fasta") +" "+ dictfile + " "+ os.path.join(outputpath, "labels.txt")
		#Execute the command and time it
		start_time = time.time()
		output = subprocess.call(command, shell=True)
		exec_time = time.time() - start_time

		testkernel = os.path.join(outputpath, "test_kernel.txt")
		#subprocess.call(["cp", "test_Kernel.txt", testkernel])
		#move massif file over to output path for analysis
		subprocess.call("mv massif.out.* "+ os.path.join(outputpath, "massif.out"), shell=True)


		print(data['name'] + "\t" + repr(exec_time))
		sys.stdout.flush()
		recordTime(os.path.join(outputpath, "times.txt"), exec_time)

		with open(os.path.join(resultspath, 'igakco', 'aggtime.txt'), 'a+') as file:
			file.write(data['name'] + "\t" + repr(exec_time))

		# with open(os.path.join(outputpath, "output.txt"), 'w') as outfile:
		# 	outfile.write(output)



def test_gakco():
	#overwrite aggregated time doc so this runs results are only ones
	f = open(os.path.join(resultspath, 'gakco', 'aggtime.txt'), 'w+')
	f.close()

	for data in datasets:
		outputpath = os.path.join(resultspath, 'gakco', data['name'].replace(".", "_"))
		trainfile = os.path.join(datapath, data['name'] + ".train.fasta")
		testfile = os.path.join(datapath, data['name'] + ".test.fasta")
		if data['name'] == 'EP300' or data['name'] == 'CTCF'or data['name'] == 'JUND'or data['name'] == 'RAD21' or data['name'] == 'SIN3A':
			dictfile = os.path.join(datapath, "dict-dna.txt")
		else:
			dictfile = os.path.join(datapath, "dict-protein.txt")


		print(data['name'])

		with open('sequences.fasta', 'w') as outfile:
			with open(trainfile) as infile:
				outfile.write(infile.read())
			with open(testfile) as infile:
				outfile.write(infile.read())

		command = ['./GaKCo', 'sequences.fasta', dictfile, os.path.join(outputpath, "labels.txt"), repr(data['g']), repr(data['g'] - data['m']), os.path.join(outputpath, "kernel.txt"), '1']
		
		#Execute the command and time it
		start_time = time.time()
		output = subprocess.check_output(command)
		exec_time = time.time() - start_time

		print(exec_time)
		sys.stdout.flush()
		recordTime(os.path.join(outputpath, "times.txt"), exec_time)

		with open(os.path.join(resultspath, 'gakco', 'aggtime.txt'), 'a+') as file:
			file.write(data['name'] + "\t" + repr(exec_time))

		with open(os.path.join(outputpath, "output.txt"), 'w') as outfile:
			outfile.write(output)


def test_gkm():
	#overwrite aggregated time doc so this runs results are only ones
	f = open(os.path.join(resultspath, 'gkm', 'aggtime.txt'), 'w+')
	f.close()

	for data in datasets:
		outputpath = os.path.join(resultspath, 'gkm', data['name'].replace(".", "_"))
		trainfile = os.path.join(datapath, data['name'] + ".train.fasta")
		testfile = os.path.join(datapath, data['name'] + ".test.fasta")
		if data['name'] == 'EP300' or data['name'] == 'CTCF'or data['name'] == 'JUND'or data['name'] == 'RAD21' or data['name'] == 'SIN3A':
			dictfile = "dna.dictionary.txt"
		else:
			dictfile = "protein.dictionary.txt"


		print(data['name'])

		with open('sequences.fasta', 'w') as outfile:
			with open(trainfile) as infile:
				outfile.write(infile.read())
			with open(testfile) as infile:
				outfile.write(infile.read())

		gkmify(trainfile, "pos.fa", "neg.fa")

		command = ["valgrind", "--tool=massif", "./gkmsvm_kernel", "-l",repr(data['g']), "-k",repr(data['g']-data['m']), "-d",repr(data['m']), "-R", "-A", dictfile, "-T", repr(20), "pos.fa", "neg.fa", os.path.join(outputpath, "kernel.txt")]
		#"-R", '-A', dictfile, '-T', repr(data['m']),


		#Execute the command and time it
		start_time = time.time()
		output = subprocess.check_output(command)
		exec_time = time.time() - start_time

		#move massif file over to output path for analysis
		subprocess.call("mv massif.out.* "+ os.path.join(outputpath, "massif.out"), shell=True)

		command = ["./gkmsvm_train", os.path.join(outputpath, "kernel.txt"), "pos.fa", "neg.fa", outputpath+"/svmtrain"]
		trainoutput = subprocess.check_output(command)
		

		gkmify(testfile, "testseq.fa", "bunkparam", True)

		command = ["./gkmsvm_classify", "-l", repr(data['g']), "-k",repr(data['g']-data['m']), "-d",repr(data['m']), "-R", "-A", dictfile, "testseq.fa", os.path.join(outputpath, "svmtrain_svseq.fa"), os.path.join(outputpath, "svmtrain_svalpha.out"), os.path.join(outputpath, "probs.txt")]
		classifyoutput = subprocess.check_output(command)


		print(exec_time)
		sys.stdout.flush()
		recordTime(os.path.join(outputpath, "times.txt"), exec_time)

		with open(os.path.join(resultspath, 'gkm', 'aggtime.txt'), 'a+') as file:
			file.write(data['name'] + "\t" + repr(exec_time))

		with open(os.path.join(outputpath, "output.txt"), 'w') as outfile:
			outfile.write(output)
			outfile.write("\n#########\n")
			outfile.write(trainoutput)
			outfile.write("\n#########\n")
			outfile.write(classifyoutput)




def main():
	if(not os.path.isdir(resultspath)):
		os.mkdir(resultspath)
		os.mkdir(os.path.join(resultspath, 'gakco'))
		os.mkdir(os.path.join(resultspath, 'igakco'))
		os.mkdir(os.path.join(resultspath, 'gkm'))
		for data in datasets:
			os.mkdir(os.path.join(resultspath, 'gakco', data['name'].replace(".", "_")))
			os.mkdir(os.path.join(resultspath, 'igakco', data['name'].replace(".", "_")))
			os.mkdir(os.path.join(resultspath, 'gkm', data['name'].replace(".", "_")))
			

	if igakco:
		test_igakco()
	if gakco:
		test_gakco()
	if gkm:
		test_gkm()


if(__name__ == "__main__"):
	main()