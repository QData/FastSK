import os
import subprocess
import time
import datetime
import sys
from sklearn.metrics import f1_score

#These flags will perform the specified tests with all the tools set to True
igakco = False
gkm = True
gakco = False

#only have one of these active at a time, each will influence the other.
#timing and memory usage only look at kernel generation and massif (memory profiler) slows down execution massively
#auc of course needs train/test time, and actually they do a cross-validation to produce it so it will take a long time as well and the time will be included in the time.
AUC_FLAG = False
MEM_FLAG = False
TIME_FLAG = True

#will ignore and use 'm' threads for comparability in memory tests.
THREAD_COUNT = 20

datapath = '/bigtemp/ec3bd/alldata'
results = '/localtmp/ec3bd/testresults'

train_percent = .9 #no longer effective to modify, however this is still the percent of data used for training in the new dna sets.

#one line command to read massif.out's peak mem consumption
"cat | grep mem_heap_B massif.out | sed -e 's/mem_heap_B=\(.*\)/\1/' | sort -g | tail -n 1"


profs = {
		# 'EP300_47878':{'name':'EP300_47878', 'g':10, 'm':5, 'c':1, 'type':'dna'},
		# 'KAT2B_45635':{'name':'KAT2B_45635', 'g':10, 'm':5, 'c':0.1, 'type':'dna'},
		# 'NR2C2_670':{'name':'NR2C2_670', 'g':10, 'm':5, 'c':1, 'type':'dna'},
		# 'TP53_41502':{'name':'TP53_41502', 'g':10, 'm':5, 'c':0.1, 'type':'dna'},
		# 'ZBTB33_48276':{'name':'ZBTB33_48276', 'g':10, 'm':5, 'c':1, 'type':'dna'},
		# 'ZZZ3_45648':{'name':'ZZZ3_45648', 'g':10, 'm':5, 'c':0.1, 'type':'dna'},
		'1.1':{'name':'1.1', 'g':7, 'm':2, 'c': .01, 'type': 'protein'},
		'1.34':{'name':'1.34', 'g':10, 'm':9, 'c': .1, 'type': 'protein'},
		'2.19':{'name':'2.19', 'g':7, 'm':6, 'c': 100, 'type': 'protein'},
		'2.31':{'name':'2.31', 'g':10, 'm':3, 'c': 10, 'type': 'protein'},
		'2.1':{'name':'2.1', 'g':10, 'm':7, 'c': 10, 'type': 'protein'},
		'2.34':{'name':'2.34', 'g':7, 'm':1, 'c': .01, 'type': 'protein'},
		'2.41':{'name':'2.41', 'g':10, 'm':4, 'c': .01, 'type': 'protein'},
		'2.8':{'name':'2.8', 'g':10, 'm':9, 'c': 10, 'type': 'protein'},
		'3.19':{'name':'3.19', 'g':8, 'm':7, 'c': .1, 'type': 'protein'},
		'3.25':{'name':'3.25', 'g':10, 'm':2, 'c': 1, 'type': 'protein'},
		'3.33':{'name':'3.33', 'g':10, 'm':5, 'c': .01, 'type': 'protein'},
		'3.50':{'name':'3.50', 'g':10, 'm':3, 'c': .01, 'type': 'protein'},
		# 'CTCF':{'name':'CTCF', 'g':10, 'm':5, 'c': 1, 'type':'dna'},
		# 'EP300':{'name':'EP300', 'g':10, 'm':5, 'c': 1, 'type':'dna'},
		# 'JUND':{'name':'JUND', 'g':10, 'm':3, 'c': 1, 'type':'dna'},  
		# 'RAD21':{'name':'RAD21', 'g':10, 'm':5, 'c': 1, 'type':'dna'},
		# 'SIN3A':{'name':'SIN3A', 'g':10, 'm':3, 'c': 1, 'type':'dna'},
		# 'sentiment':{'name':'sentiment', 'g':8, 'm': 4, 'c':1, 'type':'text'}
		}

def main():
	test_set = profs.keys()
	
	if len(sys.argv) > 1:
		test_set = []
		for i in range(1,len(sys.argv)):
			test_set.append(sys.argv[i])

	if igakco:
		test_igakco(test_set)
	if gkm:
		test_gkm(test_set)
	if gakco:
		test_gakco(test_set)


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


def to_fasta(pos, neg, trainfile, testfile):
	with open(pos, 'r') as file:
		poslines = file.readlines()
	with open(neg, 'r') as file:
		neglines = file.readlines()

	poslines = [line.strip().split() for line in poslines]
	neglines = [line.strip().split() for line in neglines]

	with open(trainfile, 'w') as file:
		for i in range(int(train_percent * len(poslines))):
			line = poslines[i]
			file.write('>'+line[1]+'\n')
			file.write(line[0] +'\n')
		for i in range(int(train_percent * len(neglines))):
			line = neglines[i]
			file.write('>0'+'\n')
			file.write(line[0] +'\n')

	with open(testfile, 'w') as file:
		for i in range(int(train_percent * len(poslines)), len(poslines)):
			line = poslines[i]
			file.write('>'+line[1]+'\n')
			file.write(line[0] +'\n')
		for i in range(int(train_percent * len(neglines)), len(neglines)):
			line = neglines[i]
			file.write('>0'+'\n')
			file.write(line[0] +'\n')

def gkm_auc_collate(datasets):
	for data in datasets:
		y = []
		prob = []
		print(data['name'])
		with open(data['name'].replace('.','_')+"/probs.txt", 'r') as f:
			with open("/bigtemp/ec3bd/alldata/"+data['name']+".test.fasta", 'r') as test:
				for line in f:
					yy = line.split('\t')
					guess = float(yy[1])
					#if guess >= 0: guess = 1
					#else: guess =0 
					prob.append(guess)
				for line in test:
					if line[0] == ">":
						if int(line[1:]) == 0: y.append(0)
						else: y.append(int(line[1:]))
		

		print(roc_auc_score(y, prob, average=None))


def test_igakco(test_set):

	dictfile = "outdict.txt"

	resultspath = os.path.join(results,'igakco')

	for data in test_set:

		if not os.path.isdir(os.path.join(resultspath,data)):
			subprocess.check_output(['mkdir','-p',os.path.join(resultspath,data)])

		outputfilepath = os.path.join(resultspath,data,'output.txt')
		outputfile = open(outputfilepath, 'w+')
		outputfile.close()	

		trainfile = os.path.join(datapath, data+".train.fasta")
		testfile = os.path.join(datapath, data+".test.fasta")

		#to_fasta(os.path.join(datapath, data+"_positive.txt"), os.path.join(datapath, data+"_negative.txt"),trainfile, testfile)

		massiffile = os.path.join(resultspath,data,'massif.out')
		#outputfile.write(repr(data)+"\n")
		# print(repr(data))
		# for g in [10,9,8,7]:
		# 	
		# 		for c in [.01,.1,1,10]:
		#for t in [profs[data]['m'], 2*profs[data]['g'], 5*profs[data]['g']]:
		for m in range(1,g):
			s=int(THREAD_COUNT/10)+1
			if TIME_FLAG:
				command = ['./iGakco','-S', repr(s), '-h','1','-p', '-g',repr(profs[data]['g']),'-m',repr(m),'-C', repr(profs[data]['c']), 
						'-t',repr(20),'-k',os.path.join(resultspath,data,'out_kernel.txt'), 
						trainfile, testfile, dictfile,os.path.join(resultspath,data,'labelout.txt')]
			elif AUC_FLAG:
				command = ['./iGakco','-S', repr(s),'-p', '-g',repr(profs[data]['g']),'-m',repr(profs[data]['m']),'-C', repr(profs[data]['c']), 
						'-t',repr(THREAD_COUNT),'-k',os.path.join(resultspath,data,'out_kernel.txt'), 
						trainfile, testfile, dictfile,os.path.join(resultspath,data,'labelout.txt')]
			elif MEM_FLAG:
				command = ['valgrind', '--tool=massif','--massif-out-file='+massiffile, './iGakco', '-S', repr(s),
					'-h','1','-p', '-g',repr(profs[data]['g']),'-m',repr(profs[data]['m']),'-C', repr(profs[data]['c']), 
					'-t',repr(THREAD_COUNT),'-k',os.path.join(resultspath,data,'out_kernel.txt'), trainfile, testfile, dictfile,os.path.join(resultspath,data,'labelout.txt')]


			#Execute the command and time it
			start_time = time.time()
			output = subprocess.check_output(command)
			exec_time = time.time() - start_time
			os.remove(dictfile)

			outputfile = open(outputfilepath, 'a')
			outputfile.write(output)
			res = data + "\t"+ "time: "+repr(exec_time)  #+" \t"+' '.join([str(g),str(m),str(c)])
			print(res)
			outputfile.write('\n'+res+'\n')
			outputfile.close()

			print("\n")


		# os.remove(dictfile)
		# command = "grep mem_heap_B massif.out | sed -e 's/mem_heap_B=\(.*\)/\1/' | sort -g | tail -n 1"
		# output = subprocess.check_output(command, shell=True).decode("utf-8")

		# outputfile.write("Peak Memory Usage (B): " + repr(output)+"\n\n")
		# print(output)

		# os.remove('massif.out')



def test_gkm(test_set):

	resultspath = os.path.join(results,'gkm')

	#BE AWARE OF THIS, change for different data types, more importantly, change global.h:dict_size definition and recompile for different data types
	

	for data in test_set:

		dictfile = '/localtmp/ec3bd/data/'+profs[data]['type']+'.dictionary.txt'
		massiffile = os.path.join(resultspath,data,'massif.out')

		if not os.path.isdir(os.path.join(resultspath,data)):
			subprocess.check_output(['mkdir','-p',os.path.join(resultspath,data)])

		outputfilepath = os.path.join(resultspath,data,'output.txt')
		outputfile = open(outputfilepath, 'w+')
		outputfile.close()	

		trainfile = os.path.join(datapath, data+".train.fasta")
		testfile = os.path.join(datapath, data+".test.fasta")

		#to_fasta(os.path.join(datapath, data+"_positive.txt"), os.path.join(datapath, data+"_negative.txt"),trainfile, testfile)


		with open('sequences.fasta', 'w') as outfile:
			with open(trainfile) as infile:
				for line in infile:
						outfile.write(line)
			with open(testfile) as infile:
				for line in infile:
						outfile.write(line)

		gkmify('sequences.fasta', 'pos.fa','neg.fa')

		
		print(repr(data))

		for t in range(profs[data]['m'], 2*profs[data]['g'], 5*profs[data]['g']):

			if AUC_FLAG or TIME_FLAG:
				command = ["./gkmsvm_kernel","-a","2", "-l",repr(profs[data]['g']), "-k",repr(profs[data]['m']), "-d",repr(profs[data]['m']), "-R", "-T", str(t),"-A",dictfile, "pos.fa", "neg.fa", os.path.join(resultspath,data,"kernel.txt")]
			if MEM_FLAG:
				command = ["valgrind", "--tool=massif","--massif-out-file="+massiffile,"./gkmsvm_kernel","-a","2", "-l",repr(profs[data]['g']), "-k",repr(profs[data]['m']), "-d",repr(profs[data]['m']), "-R","-A",dictfile, "-T", str(profs[data]['m']), "pos.fa", "neg.fa", os.path.join(resultspath,data,"kernel.txt")]
		
			#Execute the command and time it
			start_time = time.time()
			output = subprocess.check_output(command)
			exec_time = time.time() - start_time

			outputfile = open(outputfilepath, 'a')
			outputfile.write("time: "+repr(exec_time) +"\n")
			print("\t" + repr(exec_time))

			if AUC_FLAG:
				command = ["./gkmsvm_train", os.path.join(resultspath,data,"kernel.txt"), "pos.fa", "neg.fa", os.path.join(resultspath,data,"svmtrain")]
				trainoutput = subprocess.check_output(command)
				

				gkmify(testfile, "testseq.fa", "bunkparam", True)

				command = ["./gkmsvm_classify", "-l", repr(profs[data]['g']), "-k",repr(profs[data]['g']-profs[data]['m']), "-d",repr(profs[data]['g']), "-R", "-A", dictfile, "testseq.fa", os.path.join(resultspath,data, "svmtrain_svseq.fa"), os.path.join(resultspath,data, "svmtrain_svalpha.out"), os.path.join(resultspath,data, "probs.txt")]
				classifyoutput = subprocess.check_output(command)


	if AUC_FLAG:
		gkm_auc_collate(test_set)


def test_gakco(test_set):

	resultspath = os.path.join(results,'gakco')

	for data in test_set:
		if not os.path.isdir(os.path.join(resultspath,data)):
			subprocess.check_output(['mkdir','-p',os.path.join(resultspath,data)])
		outputpath = os.path.join(resultspath, data)
		trainfile = os.path.join(datapath, data + ".train.fasta")
		testfile = os.path.join(datapath, data + ".test.fasta")
		
		dictfile = '/localtmp/ec3bd/data/'+profs[data]['type']+'.dictionary.txt'


		print(data['name'])

		with open('sequences.fasta', 'w') as outfile:
			with open(trainfile) as infile:
				outfile.write(infile.read())
			with open(testfile) as infile:
				outfile.write(infile.read())

		if MEM_FLAG:
			command = ['valgrind', '--tool=massif', './GaKCo', trainfile, dictfile, os.path.join(outputpath, "labels.txt"), repr(profs[data]['g']), repr(profs[data]['g'] - profs[data]['m']), os.path.join(outputpath, "kernel.txt"), '1']
		elif AUC_FLAG or TIME_FLAG:
			command = ['./GaKCo', trainfile, dictfile, os.path.join(outputpath, "labels.txt"), repr(profs[data]['g']), repr(profs[data]['g'] - profs[data]['m']), os.path.join(outputpath, "kernel.txt"), '1']
		#Execute the command and time it
		start_time = time.time()
		output = subprocess.check_output(command)
		exec_time = time.time() - start_time

		print(exec_time)
		sys.stdout.flush()
		#recordTime(os.path.join(outputpath, "times.txt"), exec_time)

		with open(os.path.join(resultspath, 'gakco', 'aggtime.txt'), 'a+') as file:
			file.write(data['name'] + "\t" + repr(exec_time))

		with open(os.path.join(outputpath, "output.txt"), 'w') as outfile:
			outfile.write(output)
			outfile.write(data['name'] + "\ttime:  " + repr(exec_time))

	


if __name__ == '__main__':
	main()