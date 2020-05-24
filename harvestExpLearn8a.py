import argparse
import nltk
import csv
import re
# import cPickle
import time
import json
# import operator
import glob
import sys
import os
from collections import Counter
import math
# import io
import numpy
import codecs

# print(nltk.__file__)
# print(numpy.__file__)
# print(argparse.__file__)
# print(csv.__file__)
# print(re.__file__)
# print(json.__file__)
# print(operator.__file__)
# print(os.__file__)
# print(codecs.__file__)

# python3 harvestExpLearn3Test2.py Clothes.csv Eyes.csv Stuffing.csv -t BuildABear.csv


DEBUG = True

##=============================================================================================
def dprint(line, s):
	if DEBUG:
		print(str(line), ' s: ',s )
		
#------------------------
def print_timing(function_name, start_time):
	timeEnd = time.time()
	print("-----------------------", function_name, " took	",	round((timeEnd-start_time), 1), " seconds\n" )
	
#------------------------
def run_harvest_orig():
	print("44 Harvester v 4.0 w/mods")
	print("44 start run_harvest_orig()")
	SPEC_HEADERS = 'abstract|title|claim|ultimate parent'
	IPC_HEADERS = 'ipc|ip class'
	CPC_HEADERS = 'cpc|cp class'
	UPC_HEADERS = 'upc|up class|us class'
	headers = (SPEC_HEADERS,IPC_HEADERS,CPC_HEADERS,UPC_HEADERS)
	
	#measure elpased time for each function
	global time_inno
	time_inno = 0.0
	timeStart_1 = time.time()
	args = parse_input()
	print_timing("parse_input()",timeStart_1)
	timeStart = time.time()
	stopwords, wnl, args = gather_stopwords(args)
	print_timing("gather_stopwords()",timeStart)

	timeStart = time.time()
	labels, trainingSets, featuresTuple, args, wnl, stopwords, headers = extract_features(args, wnl, stopwords, headers)
	print("63 after extract_features() trainingSets :", trainingSets)
	print_timing("extract_features()",timeStart)
	
	timeStart = time.time()
	args, labels, headers, wnl, stopwords = training(labels, trainingSets, args, featuresTuple, headers, wnl, stopwords)
	print_timing("training()",timeStart)

	timeStart = time.time()
	args, labels = classify_results(args, labels)
	print_timing("classify_results()",timeStart)

	timeStart = time.time()
	classify_target_file(args, headers, labels, wnl, stopwords)
	print_timing("classify_target_file()",timeStart)

	timeEnd = time.time()
	print("\n79 Total inno–clean_pc() processing time = ", round((time_inno), 3)  )
	print("\n80 Total harvest processing time = ", round((timeEnd-timeStart_1), 1)	)
	print("81 Reached the end of harvest...")
	
##---------------------------------------------------------------------
def parse_input():
#	print("50 start parse_input()")
	parser = argparse.ArgumentParser(description='Harvest features and training NBC from csv files', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('trainingSets', nargs='*', help='List of training csv files')

	parser.add_argument('-pickleJar', '-p', default='extract.pkl', help='Data file to store results of extraction')
	#parser.add_argument('-features', '-f', default='150', type=int, help='No. of features per label')
	parser.add_argument('-specFeatures', '-sf', default='150', type=int, help='No. of features in patent specifications per label')
	#parser.add_argument('-classFeatures', '-cf', default='30', type=int, help='No. of classification features per label')
	parser.add_argument('-ipcFeatures', '-if', default='30', type=int, help='No. of classification features per label')
	parser.add_argument('-cpcFeatures', '-cf', default='30', type=int, help='No. of classification features per label')
	parser.add_argument('-upcFeatures', '-uf', default='30', type=int, help='No. of classification features per label')
	parser.add_argument('-result', '-r', default='classifier.csv', help='Results csv file showing probabilities of each feature to each label')
	parser.add_argument('-appendTrain', '-a', action='store_true', help='Training data will just be appended (not learned), and restart the training on all files')
	parser.add_argument('-learn', '-l', action='store_true', help='Csv file will be used as predictive learning data to modify the vectors and envelope files')
	parser.add_argument('-stopwords', '-s', default='patStop.txt', help='Text file containing stop words to exclude')
	parser.add_argument('-targetFile', '-t', help='Selecting will cause program to classify the target csv file, that is specified after the argument -t, e.g. -t target.csv')
	parser.add_argument('-correctWeight', '-cw', default='3', type=float, help='Weight for correct answer on vector (%)')
	parser.add_argument('-wrongWeight', '-ww', default='15', type=float, help='Weight for wrong answer on vector (%)')
	parser.add_argument('-shrinkCorrectWeight', '-scw', default='20', type=float, help='Weight for shrinking the envelope if answer is right (out of 1000)')
	parser.add_argument('-shrinkWrongWeight', '-sww', default='200', type=float, help='Weight for shrinking the envelope if answer is wrong (out of 1000)')
	arguments = parser.parse_args()
#	print("71 end parse_input()")
	return arguments

##---------------------------------------------------------------------
def inno_clean_pc(classString, classType):
	#Clean up Innography's classification data
	global time_inno
	time_start_icp = time.time()
	cleanClassTerms=[]
	classString = re.sub(r'\s?IPC_(.*?)', r'|\1',classString)
	classString = re.sub(r'^\|(.*)',r'\1',classString)
	patClasses = classString.split('|')
	badSyn = re.search(r'\w{4}\s\d{1,3}',classString)
	for patClass in patClasses:
		patClass = patClass.strip()
		if re.search(r'\w{4}\s\d{1,3}',patClass) is not None:
			patClass = re.sub(r'(\w{4}\s\d{1,3}$)',r'\1/000',patClass)
			ipc4 = re.sub(r'\s?(\w{4})\s\d{1,3}/\d{1,3}[0\s]*',r'\1',patClass)
			group = re.sub(r'\s?\w{4}\s(\d{1,3})/\d{1,3}[0\s]*',r'\1',patClass)
			subgroup = re.sub(r'\s?\w{4}\s\d{1,3}/(\d{1,3})[0\s]*',r'\1',patClass)
			patClass = ipc4+group.zfill(3)+subgroup.zfill(3)+"00"
			del ipc4, group, subgroup
		patClass=classType+"_"+patClass
		cleanClassTerms.append(patClass)
	time_stop_icp = time.time()
	time_inno = time_inno + (time_stop_icp - time_start_icp)
	return cleanClassTerms


##---------------------------------------------------------------------
def gather_stopwords(args):
#	print("95 start gather_stopwords()")
	#open stopwords file
	dict = vars(args)
	stop = dict["stopwords"]
	print("Stopwords file: ", stop)
	stopwords_local = nltk.corpus.stopwords.words('english')
	wnl_local = nltk.stem.wordnet.WordNetLemmatizer()
	stopwords_local = [wnl_local.lemmatize(w) for w in stopwords_local]
	with codecs.open(args.stopwords, 'r', encoding='utf-8') as p_stop:
		pStop = p_stop.read()
		pStops = [w.lower() for w in nltk.wordpunct_tokenize(pStop)]
		pStops = [wnl_local.lemmatize(w) for w in pStops]
	stopwords_local = stopwords_local + pStops

	del p_stop, pStop, pStops
#	print("137 end gather_stopwords()")
	return stopwords_local, wnl_local, args

##---------------------------------------------------------------------
def extract_features(args, wnl, stopwords, headers):
	print("142 start extract_features()")
	##### Constants #####
	SPEC_HEADERS = headers[0]
	IPC_HEADERS = headers[1]
	CPC_HEADERS = headers[2]
	UPC_HEADERS = headers[3]
	
	# used in extract_features()

	#patent_spec_features = []
	#patent_class_features = []
	allFeatures = []
	minSpecFeatures = args.specFeatures
	#minClassFeatures = args.classFeatures
	minIpcFeatures = args.ipcFeatures
	minCpcFeatures = args.cpcFeatures
	minUpcFeatures = args.upcFeatures
	featuresTuple = (minSpecFeatures, minIpcFeatures, minCpcFeatures, minUpcFeatures)
	trainingSets_double_list = args.trainingSets
	#print('163 trainingSets_double_list', trainingSets_double_list) # OK
	trainingSets_single_list = trainingSets_double_list[0]
	#print('165 trainingSets_single_list', trainingSets_single_list) # OK
	
	labels = []

	# check target file before processing
	# DAW mod 3/2/20
	targetFile = args.targetFile
	if targetFile:
		if (".csv") not in targetFile:
			#print("Target file, %s, must be a csv file" % (args.targetFile[0]))
			sys.exit(0)

	#extracting features
	#print("trainingSets_single_list: ", trainingSets_single_list)
#	 trainingSets_single_list = trainingSets[0]
	for trainingSet in trainingSets_single_list:
		#print("trainingSet: ", trainingSet)
	
		#	 label = trainingSetLabels[args.trainingSets.index(trainingSet)]
	# extract label from csv file, else use label, or use directory
	# DAW 3/2/20
#		 label = re.sub('.csv', '', os.path.basename(trainingSet))
		label = re.sub(r'(\w?).csv',r'\1',trainingSet)
		labels.append(label)
		if re.search('/$', trainingSet):
			labels.pop()
			trainingLists = os.listdir(trainingSet)
			trainingLists.sort(key=lambda f: os.path.splitext(f)[1])
			for trainingList in trainingLists:
				if (".csv") in trainingList: trainingSets_single_list.append("./"+trainingSet+trainingList)
				elif ("labels.txt") in trainingList:
					#with open ("./"+trainingSet+"labels.txt", 'r') as labelText:
					with codecs.open ("./"+trainingSet+"labels.txt", 'r', encoding='utf-8') as labelText:
						for labelData in labelText:
							labelData = labelData.replace("\n","")
							if labelData == "": continue
							trainingSets.append(labelData)
			continue
		elif (".csv") not in trainingSet:
			try:
				JsonDict = open(trainingSet+"_spec.txt")
			except IOError:
				#print("Label \"%s\" is not a trained label" % (trainingSet))
				#print("Available trained labels are:")
				for temp in glob.glob("./*_spec.txt"):
					temp = re.sub(r'(\w?)_spec.txt',r'\1',temp)
					#print(temp[2:])
				sys.exit(0)
			else:
				memorySpec = json.load(JsonDict)
	#			 JsonDict = open(trainingSet+"_class.txt")
	#			 memoryClass = json.load(JsonDict)
				JsonDict = open(trainingSet+"_ipc.txt")
				memoryIPC = json.load(JsonDict)
				JsonDict = open(trainingSet+"_cpc.txt")
				memoryCPC = json.load(JsonDict)
				JsonDict = open(trainingSet+"_upc.txt")
				memoryUPC = json.load(JsonDict)
				JsonDict.close()
				print("Found %s's feature files, skipping extraction" % (trainingSet))
		elif args.learn:
			print("Learning %s skipping extraction" % (trainingSet))
		else:

	# extract terms from csv file
			specFields = []
			ipcFields = []
			cpcFields = []
			upcFields = []
	#		 classificationFields = []
			specTerms = []
	#		 classTerms = []
			ipcTerms = []
			cpcTerms = []
			upcTerms = []
			specTf = Counter()
			ipcTf = Counter()
			upcTf = Counter()
			cpcTf = Counter()
			tf = {}
			memorySpec = {}
	#		 memoryClass = {}
			memoryIPC = {}
			memoryCPC = {}
			memoryUPC = {}
		
			#with open (trainingSet, 'rU') as train_FH:
			with codecs.open (trainingSet, 'rU', encoding='utf-8') as train_FH:
				#if 1:
				#train_FH = io.open(trainingSet, 'rU', encoding='ISO-8859-1')
				print("Extracting %s" %(trainingSet))
				numSpecRows = 0
	#Actual rows with PC rows is numI/U/CpcRows + numSpecRows
				numIpcRows = 0
				numCpcRows = 0
				numUpcRows = 0
				trainCsv = csv.DictReader (train_FH)
				#trainCsv = UnicodeCsvReader (train_FH)
				#trainCsv = csv.reader (train_FH)


	#Segment search terms, text and patent class
				for fieldname in trainCsv.fieldnames:
					if re.search(SPEC_HEADERS,fieldname, re.I):
						specFields.append(fieldname)
					elif re.search(IPC_HEADERS,fieldname, re.I):
						ipcFields.append(fieldname)
					elif re.search(CPC_HEADERS,fieldname, re.I):
						cpcFields.append(fieldname)
					elif re.search(UPC_HEADERS,fieldname, re.I):
						upcFields.append(fieldname)
					
				for row in trainCsv:
					rowTerms=[]
					numSpecRows += 1
						
	#Gather specs terms
					for spec in specFields:
					
						#row[spec] = row[spec].encode('ascii','ignore')
					
						#print "spec %s: %s" %(spec,row[spec])

						#row[spec] = row[spec].encode('ascii','replace')
						#temp = row[spec]
						#temp.decode('utf-8')
					
						words = [wnl.lemmatize(w) for w in nltk.wordpunct_tokenize(row[spec])]
						#print words
						for word in words:
							word = word.lower()
							if word not in stopwords:
								if word.isalpha():
									word="spec_"+word
									rowTerms.append(word)

					tf = Counter(rowTerms)

	#check if no specs
					if rowTerms:
						maxTf = tf.most_common(1)[0][1]
					else:
						maxTf = 1
						
	#normalize for length of specs

	#				 print "pre-tf is %s" %tf
					for (term,count) in tf.items():
						count = count/(1.0*maxTf)
						#count = 0.5 + ((0.5*count)/maxTf)
						tf[term]=0
						tf.update({term:count})
					specTf += tf
	#				 print "post-tf is %s" %tf
	#				 print "max tf is %i" %maxTf
	#				 print "spec tf is %s" %specTf
	
	#Get Boolean TF
					rowTerms = list(set(rowTerms))
					specTerms.extend(rowTerms)
					rowTerms=[]

	#Gather classifications
					for classCol in upcFields:
						classString = row[classCol]
						if classString=="":
							numUpcRows -= 1
							break
						rowTerms = inno_clean_pc(classString,"upc")
					if rowTerms:
						tf = Counter(rowTerms)
						maxTf = tf.most_common(1)[0][1]
						for (term,count) in tf.items():
							#count = 0.5+ ((0.5*count)/maxTf)
							count = count/(1.0*maxTf)
							tf[term]=0
							tf.update({term:count})
						upcTf += tf
					rowTerms = list(set(rowTerms))
					upcTerms.extend(rowTerms)
					rowTerms=[]

					for classCol in cpcFields:
						classString = row[classCol]
						if classString=="":
							numCpcRows -= 1
							break
						rowTerms = inno_clean_pc(classString,"cpc")
					if rowTerms:
						tf = Counter(rowTerms)
						maxTf = tf.most_common(1)[0][1]
						for (term,count) in tf.items():
							#count = 0.5+ ((0.5*count)/maxTf)
							count = count/(1.0*maxTf)
							tf[term]=0
							tf.update({term:count})
						cpcTf += tf
					rowTerms = list(set(rowTerms))
					cpcTerms.extend(rowTerms)
					rowTerms=[]

					for classCol in ipcFields:
						classString = row[classCol]
						if classString=="":
							numIpcRows -= 1
							break
						rowTerms = inno_clean_pc(classString,"ipc")
					if rowTerms:
						tf = Counter(rowTerms)
						maxTf = tf.most_common(1)[0][1]
						for (term,count) in tf.items():
							#count = 0.5+ ((0.5*count)/maxTf)
							count = count/(1.0*maxTf)
							tf[term]=0
							tf.update({term:count})
						ipcTf += tf
					rowTerms = list(set(rowTerms))
					ipcTerms.extend(rowTerms)
					rowTerms=[]

	# Gathering boolean term frequency (or document frequency variable)
	#		 uniqueClassTerms = nltk.FreqDist(classTerms)
			uniqueSpecTerms = nltk.FreqDist(specTerms)
			uniqueIpcTerms = nltk.FreqDist(ipcTerms)
			uniqueCpcTerms = nltk.FreqDist(cpcTerms)
			uniqueUpcTerms = nltk.FreqDist(upcTerms)

	# if no CPC, IPC, UPC columns then set samples to zero
			if len(upcFields)==0: numUpcRows = -numSpecRows
			if len(ipcFields)==0: numIpcRows = -numSpecRows
			if len(cpcFields)==0: numCpcRows = -numSpecRows

	# get average TF instead of using total TF
			for (term, count) in specTf.items():
				count = count/numSpecRows
				specTf[term] = 0
				specTf.update({term:count})
			for (term, count) in upcTf.items():
				count = count/(numUpcRows + numSpecRows)
				upcTf[term] = 0
				upcTf.update({term:count})
			for (term, count) in cpcTf.items():
				count = count/(numCpcRows + numSpecRows)
				cpcTf[term] = 0
				cpcTf.update({term:count})
			for (term, count) in ipcTf.items():
				count = count/(numIpcRows + numSpecRows)
				ipcTf[term] = 0
				ipcTf.update({term:count})
	#print "spectf %s" %specTf

	#print "Append train: %s" % args.appendTrain
	# if append flag is set append terms to Dictionary of memorySpec or memoryClass
	#		if args.appendTrain or args.learn:
			if args.appendTrain:
				print("Appending terms")
	# try to load memorySpec and memoryClass
				try:
					JsonSpecFH = open(label+"_spec.txt")
				except IOError:
					print("Cannot find \"%s_spec.txt\", \"%s\" is probably not a trained label" % (label,label))
					sys.exit(0)
				else:
					memorySpec = json.load(JsonSpecFH)
					JsonSpecFH.close()
	#				 with open(label+"_class.txt") as JsonClassFH:
	#					 memoryClass = json.load(JsonClassFH)
					with open(label+"_ipc.txt") as JsonClassFH:
						memoryIPC = json.load(JsonClassFH)
					with open(label+"_cpc.txt") as JsonClassFH:
						memoryCPC = json.load(JsonClassFH)
					with open(label+"_upc.txt") as JsonClassFH:
						memoryUPC = json.load(JsonClassFH)

				for term, df in uniqueSpecTerms.iteritems():
					tf = specTf[term]
					oldSample = memorySpec["samples"]
					tf *= numSpecRows
					if term in memorySpec:
						(oldDf,oldTf) = memorySpec[term]
						df += oldDf
						#tf *= numSpecRows
						oldTf *= oldSample
						tf += oldTf
					tf /= (numSpecRows + oldSample)
					memorySpec.update({term:(df,tf)})

				for term, df in uniqueIpcTerms.iteritems():
					tf = ipcTf[term]
					oldSample = memoryIPC["samples"]
					tf *= (numSpecRows + numIpcRows)
					if term in memoryIPC:
						(oldDf,oldTf) = memoryIPC[term]
						df += oldDf
						oldTf *= oldSample
						tf += oldTf
					tf /= (numSpecRows + numIpcRows + oldSample)
					memoryIPC.update({term:(df,tf)})

				for term, count in uniqueCpcTerms.iteritems():
					tf = cpcTf[term]
					oldSample = memoryCPC["samples"]
					tf *= (numSpecRows + numCpcRows)
					if term in memoryCPC:
						(oldDf,oldTf) = memoryCPC[term]
						df += oldDf
						oldTf *= oldSample
						tf += oldTf
					tf /= (numSpecRows + numCpcRows + oldSample)
					memoryCPC.update({term:(df,tf)})

				for term, count in uniqueUpcTerms.iteritems():
					tf = upcTf[term]
					oldSample = memoryUPC["samples"]
					tf *= (numSpecRows + numUpcRows)
					if term in memoryUPC:
						(oldDf,oldTf) = memoryUPC[term]
						df += oldDf
						oldTf *= oldSample
						tf += oldTf
					tf /= (numSpecRows + numUpcRows + oldSample)
					memoryUPC.update({term:(df,tf)})

				memorySpec["samples"] = memorySpec["samples"] + numSpecRows
				memoryUPC["samples"] = memoryUPC["samples"] + numSpecRows + numUpcRows
				memoryCPC["samples"] = memoryCPC["samples"] + numSpecRows + numCpcRows
				memoryIPC["samples"] = memoryIPC["samples"] + numSpecRows + numIpcRows

	# if append flag is not set then create new dictionary and index
			elif args.learn is not None:
				print("Not appending")
				for term, df in uniqueSpecTerms.items():
					memorySpec.update({term:(df,specTf[term])})
				for term, df in uniqueIpcTerms.items():
					memoryIPC.update({term:(df,ipcTf[term])})
				for term, df in uniqueCpcTerms.items():
					memoryCPC.update({term:(df,cpcTf[term])})
				for term, df in uniqueUpcTerms.items():
					memoryUPC.update({term:(df,upcTf[term])})
	#create sample size data
				memorySpec.update({"samples":numSpecRows})
				memoryIPC.update({"samples":(numIpcRows+numSpecRows)})
				memoryUPC.update({"samples":(numUpcRows+numSpecRows)})
				memoryCPC.update({"samples":(numCpcRows+numSpecRows)})

	#		 print "memory %s" %memorySpec
	#		 print "key %s" %memorySpec.keys()
	#		 print "value %s" %memorySpec.values()
	#print memoryIPC

	# save features files but save it as temp if learning.

	#		 if args.learn:
	#			 with open(label+"_spec_temp.txt", 'w') as memOut:
	#				 json.dump(memorySpec, memOut)
	#			 with open(label+"_ipc_temp.txt", 'w') as memOut:
	#				 json.dump(memoryIPC, memOut)
	#			 with open(label+"_cpc_temp.txt", 'w') as memOut:
	#				 json.dump(memoryCPC, memOut)
	#			 with open(label+"_upc_temp.txt", 'w') as memOut:
	#				 json.dump(memoryUPC, memOut)
	#		 else:
			if args.learn is not None:
				with open(label+"_spec.txt", 'w') as memOut:
					json.dump(memorySpec, memOut)
				with open(label+"_ipc.txt", 'w') as memOut:
					json.dump(memoryIPC, memOut)
				with open(label+"_cpc.txt", 'w') as memOut:
					json.dump(memoryCPC, memOut)
				with open(label+"_upc.txt", 'w') as memOut:
					json.dump(memoryUPC, memOut)

			del spec, word, words, train_FH, fieldname, row, trainCsv, specTerms, specFields, upcFields, ipcFields, cpcFields, uniqueSpecTerms, rowTerms, term, numSpecRows, specTf, upcTf, ipcTf, cpcTf, tf, df
	return labels, trainingSets_single_list, featuresTuple, args, wnl, stopwords, headers
	print("538 end extract_features()")

##-------------------------------------------------------------------------------------------------------------
def training(labels, trainingSets, args, featuresTuple, headers, wnl, stopwords):
	print("519 start training()")
#Training documentation
	SPEC_HEADERS = headers[0]
	IPC_HEADERS = headers[1]
	CPC_HEADERS = headers[2]
	UPC_HEADERS = headers[3]

#save raw classification data
	numSamples = {}
	scoreCardSpec = {}
	scoreCardIPC = {}
	scoreCardUPC = {}
	scoreCardCPC = {}
	trainingVector = {}
	trainingLabels = list(labels)
	#print('534 trainingLabels: ', trainingLabels)
	
	minSpecFeatures = featuresTuple[0]
	minIpcFeatures = featuresTuple[1]
	minCpcFeatures = featuresTuple[2]
	minUpcFeatures = featuresTuple[3]
	#print("539 in training() trainingSets: ", trainingSets)
	# Check if vector file is created and there is no ".csv" in the argument, remove those from evaluation
	for trainingSet in trainingSets:
		print("582 trainingSets: ", trainingSets)
		print("583 trainingSet: ", trainingSet)
		if (".csv") not in trainingSet:
			try:
				JsonDict = open(trainingSet+"_vector.txt")
			except IOError:
				print("Label \"%s\" does not have a vector file, creating one." % (trainingSet))
				#print "Available trained labels are:"
				#for temp in glob.glob("./*_spec.txt"):
				#	 temp = re.sub(r'(\w?)_spec.txt',r'\1',temp)
				#	 print temp[2:]
				#sys.exit(0)
			else:
				#			 memorySpec = json.load(JsonDict)
				#			 JsonDict = open(trainingSet+"_class.txt")
				#			 memoryClass = json.load(JsonDict)
				#JsonDict = open(trainingSet+"_ipc.txt")
				#memoryIPC = json.load(JsonDict)
				#JsonDict = open(trainingSet+"_cpc.txt")
				#memoryCPC = json.load(JsonDict)
				#JsonDict = open(trainingSet+"_upc.txt")
				#memoryUPC = json.load(JsonDict)
				#JsonDict.close()
				trainingSet = re.sub(r'(\w?).csv',r'\1',trainingSet)
				trainingLabels.remove(trainingSet)
				#print("569 Found %s's vector file, skipping training." % (trainingSet))
	
		#print('568 trainingLabels: ', trainingLabels)
		for traininglabel in trainingLabels:
			print('610 trainingLabel: ', traininglabel)
		# Take csv files to undergo predictive learning

		# Learning
			if args.learn:
				print("Learning from %s." %(traininglabel))
		
				with open (traininglabel+".csv", 'rU') as learn_FH:
					learnCsv = csv.DictReader (learn_FH)
					specFields=[]
					ipcFields=[]
					cpcFields=[]
					upcFields=[]
					#rowTerms=[]
					learnHeader=[]
					weights={}
					#Segment search terms, text and patent class
					for fieldname in learnCsv.fieldnames:
						if re.search(SPEC_HEADERS,fieldname, re.I):
							specFields.append(fieldname)
						elif re.search(IPC_HEADERS,fieldname, re.I):
							ipcFields.append(fieldname)
						elif re.search(CPC_HEADERS,fieldname, re.I):
							cpcFields.append(fieldname)
						elif re.search(UPC_HEADERS,fieldname, re.I):
							upcFields.append(fieldname)
						learnHeader.append(fieldname)


					specOldFeatureCount = 0
					#specOldFeatureMin = float("inf")
					#specLowestFeature = ""
					ipcOldFeatureCount = 0
					#ipcOldFeatureMin = float("inf")
					#ipcLowestFeature = ""
					upcOldFeatureCount = 0
					#upcOldFeatureMin = float("inf")
					#upcLowestFeature = ""
					cpcOldFeatureCount = 0
					#cpcOldFeatureMin = float("inf")
					#cpcLowestFeature = ""
					with open(traininglabel+"_vector.txt") as JsonDict:
						trainingVector = json.load(JsonDict)
						features = trainingVector.keys()
					with open(traininglabel+"_spec.txt") as JsonDict:
						specFeatureData = json.load(JsonDict)
						specsample = specFeatureData.pop("samples")
						for feature in features:
							if feature in specFeatureData:
								specOldFeatureCount += 1
								#if specFeatureData[feature][1] < specOldFeatureMin:
								#	 specOldFeatureMin = specFeatureData[feature][1]
								#	 specLowestFeature = feature
						#print "for feature %s count is %s min is %s at %s" %(feature,specOldFeatureCount,specOldFeatureMin,specLowestFeature)
					with open(traininglabel+"_ipc.txt") as JsonDict:
						ipcFeatureData = json.load(JsonDict)
						ipcsample = ipcFeatureData.pop("samples")
						for feature in features:
							if feature in ipcFeatureData:
								ipcOldFeatureCount += 1
								#if ipcFeatureData[feature][1] < ipcOldFeatureMin:
								#	 ipcOldFeatureMin = ipcFeatureData[feature][1]
								#	 ipcLowestFeature = feature
						#print "for feature %s count is %s min is %s at %s" %(feature,ipcOldFeatureCount,ipcOldFeatureMin,ipcLowestFeature)
					with open(traininglabel+"_upc.txt") as JsonDict:
						upcFeatureData = json.load(JsonDict)
						upcsample = upcFeatureData.pop("samples")
						for feature in features:
							if feature in upcFeatureData:
								upcOldFeatureCount += 1
								#if upcFeatureData[feature][1] < upcOldFeatureMin:
								#	 upcOldFeatureMin = upcFeatureData[feature][1]
								#	 upcLowestFeature = feature
						#print "for feature %s count is %s min is %s at %s" %(feature,upcOldFeatureCount,upcOldFeatureMin,upcLowestFeature)
					with open(traininglabel+"_cpc.txt") as JsonDict:
						cpcFeatureData = json.load(JsonDict)
						cpcsample = cpcFeatureData.pop("samples")
						for feature in features:
							if feature in cpcFeatureData:
								cpcOldFeatureCount += 1
								#if cpcFeatureData[feature][1] < cpcOldFeatureMin:
								#	 cpcOldFeatureMin = cpcFeatureData[feature][1]
								#	 cpcLowestFeature = feature

					# Open new appended feature data
					#with open(traininglabel+"_spec_temp.txt") as JsonDict:
					#	 featureData = json.load(JsonDict)
					#	 sample = featureData.pop("samples")
					#with open(traininglabel+"_ipc_temp.txt") as JsonDict:
					#	 featureData.update(json.load(JsonDict))
					#	 ipcsample = featureData.pop("samples")
					#with open(traininglabel+"_upc_temp.txt") as JsonDict:
					#	 featureData.update(json.load(JsonDict))
					#	 upcsample = featureData.pop("samples")
					#with open(traininglabel+"_cpc_temp.txt") as JsonDict:
					#	 featureData.update(json.load(JsonDict))
					#	 cpcsample = featureData.pop("samples")

		#			 # Calculate weight dictionary
		#			 for label in labels:
		#				 with open(label+"_vector.txt") as JsonDict:
		#					 trainingVector = json.load(JsonDict)
		#
		#				 with open(label+"_spec.txt") as JsonDict:
		#					 weightFeatureData = json.load(JsonDict)
		#				 weightFeatureData.pop("samples")
		#
		#			 # Find common key between the two dictionaries
		#				 for i in range(len(trainingVector)):
		#					 if trainingVector.keys()[i] in weightFeatureData.keys():
		#						 weights[label] = trainingVector[trainingVector.keys()[i]] / weightFeatureData[trainingVector.keys()[i]][1]
		#						 break
		#
		#			del weightFeatureData

					# Evaluate each row
					for row in learnCsv:
						elementTerms=[]
						rowTerms=[]
						newFeatures={}
						hasSpec=0.0
						hasIPC=0.0
						hasCPC=0.0
						hasUPC=0.0
						oldKeys=[]

						#Gather specs terms
						for spec in specFields:
							words = [wnl.lemmatize(w) for w in nltk.wordpunct_tokenize(row[spec])]
							for word in words:
								word = word.lower()
								if word not in stopwords:
									if word.isalpha():
										word="spec_"+word
										rowTerms.append(word)
										hasSpec=1.0

						#Gather classifications
						for classCol in upcFields:
							classString = row[classCol]
							if classString=="": break
							elementTerms = inno_clean_pc(classString,"upc")
							rowTerms.extend(elementTerms)
							hasUPC=1.0
						elementTerms=[]

						for classCol in ipcFields:
							classString = row[classCol]
							if classString=="": break
							elementTerms = inno_clean_pc(classString,"ipc")
							rowTerms.extend(elementTerms)
							hasIPC=1.0
						elementTerms=[]
					
						for classCol in cpcFields:
							classString = row[classCol]
							if classString=="": break
							elementTerms = inno_clean_pc(classString,"cpc")
							rowTerms.extend(elementTerms)
							hasCPC=1.0
						elementTerms=[]

						#Evaluate the labels one at a time, Record in result buffer
						euclideanHighScore = float("inf")
						cosineHighScore = -1.0
						euclideanLabelClass = ""
						cosineLabelClass = ""

						tf = Counter(rowTerms)
						#rowCounters = Counter(rowTerms)
						#				 rowCounters = tf
					
						#check if no terms in row
						if rowTerms:
							maxTf = tf.most_common(1)[0][1] * 1.0
						else:
							maxTf = 1.0

						newFeatures.update(tf) #IPCs will not get up beyond specs, might use tf only, and not get the most common?

						# update samples
						specsample += hasSpec
						upcsample += hasUPC
						cpcsample += hasCPC
						ipcsample += hasIPC

		#				 print "new FeaTures %s" %newFeatures
		#				 print "old sPecFeautREData %s" %specFeatureData

		# adjust old feature data with new row set if they are found in the old set
						for featureKey in specFeatureData.keys():
							oldKeys.append(featureKey)
							(count,tf) = specFeatureData[featureKey]
		#					 print "is %s = %s?" %(featureKey,newFeatures.keys())
							if featureKey in newFeatures.keys():
		#						 print "yes"
								#	 (count,tf) = specFeatureData[newFeaturesKey]
								#print "previous count is %s, tf is %s for %s sample is %s" %(count, tf, featureKey, specsample)
								count += newFeatures[featureKey]
								tf = (((tf * (specsample-1.0)) + (newFeatures[featureKey]/maxTf)) /specsample)
								#print "after count is %s, tf is %s for %s" %(count, tf, featureKey)
								specFeatureData[featureKey] = (count,tf)
							else:
		#						 print "no"
								#(count,tf) = specFeatureData[newFeaturesKey]
								tf *= ((specsample-1.0)/specsample)
								#print "not found %s, reduce Tf to %s" %(featureKey,tf)
								specFeatureData[featureKey] = (count,tf)
		#				 print "modern specfeature data %s" %specFeatureData
							
						for featureKey in ipcFeatureData.keys():
							oldKeys.append(featureKey)
							(count,tf) = ipcFeatureData[featureKey]
							
							if featureKey in newFeatures.keys():
								#	 (count,tf) = specFeatureData[newFeaturesKey]
								#print "previous count is %s, tf is %s for %s sample is %s" %(count, tf, featureKey, specsample)
								count += newFeatures[featureKey]
								tf = (((tf * (specsample-1.0)) + (newFeatures[featureKey]/maxTf)) /specsample)
								#print "after count is %s, tf is %s for %s" %(count, tf, featureKey)
								ipcFeatureData[featureKey] = (count,tf)
							else:
								#(count,tf) = specFeatureData[newFeaturesKey]
								tf *= ((specsample-1.0)/specsample)
								#print "not found %s, reduce Tf to %s" %(featureKey,tf)
								ipcFeatureData[featureKey] = (count,tf)


						for featureKey in upcFeatureData.keys():
							oldKeys.append(featureKey)
							(count,tf) = upcFeatureData[featureKey]
			
							if featureKey in newFeatures.keys():
								#print "previous count is %s, tf is %s for %s sample is %s" %(count, tf, featureKey, specsample)
								count += newFeatures[featureKey]
								tf = (((tf * (specsample-1.0)) + (newFeatures[featureKey]/maxTf)) /specsample)
								#print "after count is %s, tf is %s for %s" %(count, tf, featureKey)
								upcFeatureData[featureKey] = (count,tf)
							else:
								#(count,tf) = specFeatureData[newFeaturesKey]
								tf *= ((specsample-1.0)/specsample)
								#print "not found %s, reduce Tf to %s" %(featureKey,tf)
								upcFeatureData[featureKey] = (count,tf)

						for featureKey in cpcFeatureData.keys():
							oldKeys.append(featureKey)
							(count,tf) = cpcFeatureData[featureKey]
			
							if featureKey in newFeatures.keys():
								#print "previous count is %s, tf is %s for %s sample is %s" %(count, tf, featureKey, specsample)
								count += newFeatures[featureKey]
								tf = (((tf * (specsample-1.0)) + (newFeatures[featureKey]/maxTf)) /specsample)
								#print "after count is %s, tf is %s for %s" %(count, tf, featureKey)
								cpcFeatureData[featureKey] = (count,tf)
							else:
								tf *= ((specsample-1.0)/specsample)
								#print "not found %s, reduce Tf to %s" %(featureKey,tf)
								cpcFeatureData[featureKey] = (count,tf)

						# Now add new feature terms to file (not training vector)
		#				 diffs = list(set(newFeatures.keys()) - set(features))
						featureDataDiffs = list(set(newFeatures.keys()) - set(oldKeys))
						#print "$$$$$$$$$$$$$$$$$$$$$$$ diffs %s" %featureDataDiffs
						for featureDataDiff in featureDataDiffs:
							count = newFeatures[featureDataDiff]
							tf = (count/maxTf)
							if "spec_" in featureDataDiff:
								tf /= specsample
		#						print "Spec for %s new tf is %s, for count %s, max tf %s and spec sample as %s" %(diff,tf,count,maxTf,specsample)
								specFeatureData.update({featureDataDiff:(count,tf)})
							elif "ipc_" in featureDataDiff:
								tf /= ipcsample
								ipcFeatureData.update({featureDataDiff:(count,tf)})
							elif "cpc_" in featureDataDiff:
								tf /= cpcsample
								cpcFeatureData.update({featureDataDiff:(count,tf)})
							elif "upc_" in featureDataDiff:
								tf /= upcsample
								upcFeatureData.update({featureDataDiff:(count,tf)})

		#				 for newFeaturesKey in newFeatures.keys():
		#					 if "spec_" in newFeaturesKey:
		#						 if newFeaturesKey in specFeatureData.keys():
		#							 pass
									#(count,tf) = specFeatureData[newFeaturesKey]
									#print "previous count is %s, tf is %s for %s sample is %s" %(count, tf, newFeaturesKey, specsample)
									#count += newFeatures[newFeaturesKey]
									#tf = (((tf * (specsample-1)) + (newFeatures[newFeaturesKey]/maxTf)) /specsample)
									#print "after count is %s, tf is %s for %s" %(count, tf, newFeaturesKey)
									#specFeatureData[newFeaturesKey] = (count,tf)
									#						 else:
		#pass
		#					 elif "ipc_" in newFeaturesKey:
		#						 pass
		#					 elif "cpc_" in newFeaturesKey:
		#						 pass
		#					 elif "upc_" in newFeaturesKey:
		#						 pass
			

		#diffs = newFeatures.keys()
		#dfs = {}

		#### Memory Suck: I guess we loaded all the specs to memory

						for label in labels:
							features =[]
							labelVector = []
							rowVector = []
					
							with open(label+"_vector.txt") as JsonDict:
								trainingVector = json.load(JsonDict)
							with open(label+"_envelope.txt") as JsonDict:
								trainingEnvelope = json.load(JsonDict)

							features = trainingVector.keys()
							labelVector = trainingVector.values()
							rowVector = [0.0]*(len(labelVector))

							for rowTerm in rowTerms:
								if rowTerm in features:
									rowVector[features.index(rowTerm)] +=1
							rowVector = [i/maxTf for i in rowVector]

		#					 trainingCommons = list(set(newFeatures.keys()) & set(trainingVector.keys()))
		#					 for trainingCommon in trainingCommons:
		#						 rowVector[trainingVector.keys().index(trainingCommon)] = newFeatures[trainingCommon]/maxTf

		#					 print "row vector is %s, for label %s" %(rowVector,label)
		#					 print "maxtf is %s" %maxTf
							# need to normalize the rowvector with that of the trainingvector, averaging and divided maxTF, equation is ((samples*trainingVector[term](can be 0))+rowTerm)/(samples+1)
							# but if you do that then the new terms will be drowned out by the large amount of old terms, maybe choose by
		#delFeatures = []

		#with open(label+"_spec_temp.txt") as JsonDict:
		#featureData = json.load(JsonDict)
		#sample = featureData.pop("samples")
		#with open(label+"_ipc_temp.txt") as JsonDict:
		#featureData.update(json.load(JsonDict))
		#ipcsample = featureData.pop("samples")
		#with open(label+"_upc_temp.txt") as JsonDict:
		#featureData.update(json.load(JsonDict))
		#upcsample = featureData.pop("samples")
		#with open(label+"_cpc_temp.txt") as JsonDict:
		#featureData.update(json.load(JsonDict))
		#cpcsample = featureData.pop("samples")
		#print "features data %s" %featureData
		#print "tf data %s" %tf
		#print "sample %s for %s" %(sample,label)
		#print "ipc sample %s" %ipcsample
		#if label == traininglabel:
		#pass
		#print "new features %s" %newFeatures
								# remove old features
								#						 print featureData
								#diff = list(set(newFeatures.keys()) - set(featureData.keys()))
								#print "diff %s" %diff
								#print "diffs %s" %diffs

		#check that they do not use the above equation when adding in new features! from learning!
		#if newFeature in featureData.keys():
									#		print "found the newFeature %s" %newFeature
									#	 delFeatures.append(newFeature)
		#print "del features %s" %delFeatures


		#else:
		#pass
		#for diff in diffs:
		#pass
		#if diff in featureData:
										# this feature is found in another label! need to deemphasize this feature in the other label
										#print "tesm %s for diff %s in label %s whole feature data is %s" %(featureData[diff][0], diff, label, featureData[diff])
										#print "feature data is %s" %featureData
										#dfs[diff] = featureData[diff][0]
										#print "feature data diff 0 %s" %featureData[diff][0]

							#					 with open(label+"_upc.txt") as JsonDict:
							#	 trainingEnvelope = json.load(JsonDict)
							#with open(label+"_ipc.txt") as JsonDict:
							#	 trainingEnvelope = json.load(JsonDict)
							#with open(label+"_cpc.txt") as JsonDict:
							#	 trainingEnvelope = json.load(JsonDict)


				
					#					 sample = memorySpec.pop("samples")
					#		 numSamples.update({traininglabel+"_spec":sample})
					#		 # Gather idf for spec features, switch feature count to tf in memorySpec
					#		 for term, (count,tf) in memorySpec.iteritems():
					#			 otherSample = 0.0
					#			 df = 0.0
					#			 for otherLabel in otherLabels:
					#				 JsonDict = open(otherLabel+"_spec.txt")
					#				 otherDoc = json.load(JsonDict)
					#				 JsonDict.close()
					#				 if term in otherDoc:
					#					 (odf,otf) = otherDoc[term]
					#					 df += odf
					#				 otherSample += otherDoc["samples"]
					#			 tfIdf = tf * (math.log(((1.5+otherSample)/(1+df))))
					#			 memorySpec[term] = (tf,tfIdf)


							euclideanScore = numpy.subtract(labelVector,rowVector)
							euclideanScore = [i ** 2 for i in euclideanScore]
							euclideanScore = sum(euclideanScore)
						
							#print "euclidean score is %s for %s with envelope of %s" %(euclideanScore,label,trainingEnvelope['euclidean'])

							cosineScore = numpy.linalg.norm(labelVector)
							cosineScore *= numpy.linalg.norm(rowVector)
		# cosine score of 0 indicates that the vector is perpendicular (1 means parallel), this escapes a div/0 error
							if cosineScore == 0:
								cosineScore = -1
							else:
								cosineScore = numpy.dot(labelVector,rowVector) / cosineScore
						
							#					 print "euclidean score is %s for %s with envelope of %s" %(euclideanScore,label,trainingEnvelope['euclidean'])
							#print "cosine score is %s for %s with envelope of %s" %(cosineScore,label,trainingEnvelope['cosine'])

						# Determine label for row
							if euclideanScore<=trainingEnvelope['euclidean']:
								if euclideanLabelClass == "":
									euclideanLabelClass=label
									euclideanHighScore =euclideanScore
								else:
									if euclideanScore<euclideanHighScore:
										euclideanLabelClass=label
										euclideanHighScore =euclideanScore
					
							if cosineScore>=trainingEnvelope['cosine']:
								if cosineLabelClass == "":
									cosineLabelClass=label
									cosineHighScore =cosineScore
								else:
									if cosineScore>cosineHighScore:
										cosineLabelClass=label
										cosineHighScore =cosineScore
	  
		# Testing for euclidean only
		#				 print "euclideanHighscore %s for %s" %(euclideanHighScore,euclideanLabelClass)
		#				 print "cosineHighscore %s for %s" %(cosineHighScore,cosineLabelClass)

						# Need correct vector regardless of guess being correct or wrong
						with open(traininglabel+"_vector.txt") as JsonDict:
							trainingVector = json.load(JsonDict)
							features = trainingVector.keys()
							labelVector = trainingVector.values()

						with open(traininglabel+"_envelope.txt") as JsonDict:
							envelope = json.load(JsonDict)

		# need to update weights for correct and wrong guesses because each row is calculated

						if euclideanLabelClass == traininglabel:
		# Guessed correctly
		#					 print "guessed correctly"

							trainingCommons = list(set(newFeatures.keys()) & set(features))
							for trainingCommon in trainingCommons:
								# intentionally do not divide by numspecrow because we do not want to average out the tf (take each row as a learning experience), else just appending data
								trainingVector[trainingCommon] += newFeatures[trainingCommon] * (args.correctWeight/100.0) / maxTf

							vectorWeight = [i ** 2 for i in trainingVector.values()]
							vectorWeight = math.sqrt((100/sum(vectorWeight)))
							for key in trainingVector.keys():
								trainingVector[key] *= vectorWeight
						
							# save the vector file
							with open(euclideanLabelClass+"_vector.txt", 'w') as memOut:
								json.dump(trainingVector, memOut)

		# change envelope by a factor
							envelope['euclidean'] -= (args.shrinkCorrectWeight/10.0)
							envelope['cosine'] += (args.shrinkCorrectWeight /1000.0)

		# throttle back shrinkage of envelope if it would exclude the original answer
							if envelope['euclidean'] < euclideanHighScore:
								#print "HOLD THE LINE!!!!!!!"
								envelope['euclidean'] = euclideanHighScore
							if envelope['cosine'] > cosineHighScore:
								envelope['cosine'] = cosineHighScore

							with open(euclideanLabelClass+"_envelope.txt", 'w') as memOut:
								json.dump(envelope, memOut)
						
						else:
		# Guessed wrongly
		#					 print "guessed wrongly: guessed %s at %s, but is actually %s" %(euclideanLabelClass,euclideanHighScore,traininglabel)
						
							trainingDiffs = list(set(newFeatures.keys()) - set(features))
							trainingCommons = list(set(newFeatures.keys()) & set(features))
						
		# enhance features that are common
							for trainingCommon in trainingCommons:
								trainingVector[trainingCommon] += newFeatures[trainingCommon] * (args.correctWeight/100.0) / maxTf

							#### we pull current spec/upc/ipc/cpc features each time we get it wrong, because each time we get it right we update the general list instead
							currentSpecFeatures = {}
							currentIpcFeatures = {}
							currentCpcFeatures = {}
							currentUpcFeatures = {}
						
							for feature in features:
								if "spec_" in feature:
									currentSpecFeatures[feature] = specFeatureData[feature][1]
								elif "ipc_" in feature:
									currentIpcFeatures[feature] = ipcFeatureData[feature][1]
								elif "cpc_" in feature:
									currentCpcFeatures[feature] = cpcFeatureData[feature][1]
								elif "upc_" in feature:
									currentUpcFeatures[feature] = upcFeatureData[feature][1]
					
						#specOldFeatureMin = [for i in specFeatureData[]
					
							# Check if feature should be added (only 1 feature per type)
							for trainingDiff in trainingDiffs:
						
								if "spec_" in trainingDiff:
							
									minimumSpec = min(currentSpecFeatures, key=currentSpecFeatures.get)
		#							 print "diff = %s" %trainingDiff
		#							 print "sepc feature data %s" %specFeatureData
		#							 print "current spec features %s" %currentSpecFeatures
		#							 print "minimum spec is %s" %minimumSpec


									# add because there is insufficient spec terms
									#if specOldFeatureCount < args.specFeatures or specFeatureData[diff][1] > specOldFeatureMin:
									if specOldFeatureCount < args.specFeatures:
										# Add spec feature
										#newVector = featureData[diff][1] ** 2
										#newWeight = math.sqrt(100/((100/(oldWeight**2))+newVector))
										##############
										#oldWeight = newWeight/oldWeight
										#for term, vector in trainingVector.iteritems():
										#	 vector = oldWeight * vector
										#	 trainingVector.update({term:vector})

										#trainingVector.update({diff:(featureData[diff][1]*newWeight)})
										#print "updated with %s" %featureData[diff][1]
										#								 print "@#$@#@# adding %s spec vector $#@$@#$@" %trainingDiff
		#								 print "args %s old feature count %s" %(args.specFeatures,specOldFeatureCount)
										trainingVector.update({trainingDiff:(newFeatures[trainingDiff]/maxTf)})
										currentSpecFeatures[trainingDiff] = newFeatures[trainingDiff]/maxTf
										#features.append(diff)	   # don't need to update features, feature set is reused for each row
		#								 print "training vector %s" %trainingVector
										specOldFeatureCount+=1
									
										# remove lowest TF term if there are too many spec terms
		#								 if specOldFeatureCount >= args.specFeatures and specFeatureData[diff][1] > specOldFeatureMin:
		#									 specOldFeatureMin = specFeatureData[diff][1]
		#									 print "#^^#@$%@#$%@#$%@#$^%@#^@#$"
		#									 print "before trianing vector %s" %trainingVector
		#									 del trainingVector[specLowestFeature]
		#									 print "lowest %s diff %s" %(specLowestFeature,diff)
		#									 print "training Vector %s" %trainingVector
											######### better investigate this ##########
											# might not be true... need to calculate the lowest feature
		#									 specLowestFeature = diff
		#									 specOldFeatureCount -=1
							
							
		#							 elif specOldFeatureCount >= args.specFeatures and specFeatureData[diff][1] >= specOldFeatureMin:
									elif specOldFeatureCount >= args.specFeatures and specFeatureData[trainingDiff][1] >= currentSpecFeatures[minimumSpec]:
										#specOldFeatureMin = specFeatureData[diff][1]
										#								 print "@@#@#@#@#@#@ Replacing %s with %s @#@#@#@#@" % (minimumSpec, trainingDiff)
										del trainingVector[minimumSpec]
										trainingVector.update({trainingDiff:(newFeatures[trainingDiff]/maxTf)})
										currentSpecFeatures[trainingDiff] = newFeatures[trainingDiff]/maxTf
										del currentSpecFeatures[minimumSpec]
										#features.remove(specLowestFeature)
									
										#minimumSpec = min(currentSpecFeatures, key=currentSpecFeatures.get)

		#								 print "lowest %s diff %s" %(minimumSpec,trainingDiff)
		#								 print "training Vector %s" %trainingVector
											######### better investigate this ##########
											# might not be true... need to calculate the lowest feature
										#specLowestFeature = diff
								
		#							 elif specOldFeatureCount >= args.specFeatures and specFeatureData[diff][1] < specOldFeatureMin:
									elif specOldFeatureCount >= args.specFeatures and specFeatureData[trainingDiff][1] < currentSpecFeatures[minimumSpec]:
										pass
										#								 print "@@#@#@#@#@#@ All full and %s is not good enough with %s @ %s @#@#@#@#@" %(trainingDiff, minimumSpec, currentSpecFeatures[minimumSpec])
		#								 print "lowest %s diff %s diff's spec %s currentspecfeatures %s at %s" %(minimumSpec,trainingDiff, specFeatureData[trainingDiff][1], currentSpecFeatures, currentSpecFeatures[minimumSpec])
		#								 print "training Vector %s" %trainingVector
										######### better investigate this ##########
										# might not be true... need to calculate the lowest feature


										# Update vector weight
										#oldWeight = newWeight
										#del newVector, newWeight, term, vector


								elif "ipc_" in trainingDiff:
									pass
									minimumIpc = min(currentIpcFeatures, key=currentIpcFeatures.get)

									if ipcOldFeatureCount < args.ipcFeatures:
										#print "@#$@#@# adding %s ipc vector $#@$@#$@" %trainingDiff
										trainingVector.update({trainingDiff:(newFeatures[trainingDiff]/maxTf)})
										currentIpcFeatures[trainingDiff] = newFeatures[trainingDiff]/maxTf
										ipcOldFeatureCount+=1

									elif ipcOldFeatureCount >= args.ipcFeatures and ipcFeatureData[trainingDiff][1] >= currentIpcFeatures[minimumIpc]:
										#print "@@#@#@#@#@#@ Replacing %s with %s @#@#@#@#@" % (minimumIpc, trainingDiff)
										del trainingVector[minimumIpc]
										trainingVector.update({trainingDiff:(newFeatures[trainingDiff]/maxTf)})
										currentIpcFeatures[trainingDiff] = newFeatures[trainingDiff]/maxTf
										del currentIpcFeatures[minimumIpc]

									elif ipcOldFeatureCount >= args.ipcFeatures and ipcFeatureData[trainingDiff][1] < currentIpcFeatures[minimumIpc]:
										pass
										#print "@@#@#@#@#@#@ All full and %s is not good enough with %s @ %s @#@#@#@#@" %(trainingDiff, minimumIpc, currentIpcFeatures[minimumIpc])

		#							 if ipcOldFeatureCount < args.ipcFeatures or ipcFeatureData[diff][1] > ipcOldFeatureMin:
		#								 trainingVector.update({diff:(newFeatures[diff]/maxTf)})
									
		#								 if ipcOldFeatureCount >= args.ipcFeatures and ipcFeatureData[diff][1] > ipcOldFeatureMin:
		#									 ipcOldFeatureMin = ipcFeatureData[diff][1]
		#									 del trainingVector[ipcLowestFeature]
		#									 print "lowest %s diff %s" %(ipcLowestFeature,diff)
		#									 ipcLowestFeature = diff
		#									 ipcOldFeatureCount -=1
									
		#								 ipcOldFeatureCount+=1

								elif "upc_" in trainingDiff:
									pass

									minimumUpc = min(currentUpcFeatures, key=currentUpcFeatures.get)
		
									if upcOldFeatureCount < args.upcFeatures:
										#print "@#$@#@# adding %s upc vector $#@$@#$@" %trainingDiff
										trainingVector.update({trainingDiff:(newFeatures[trainingDiff]/maxTf)})
										currentUpcFeatures[trainingDiff] = newFeatures[trainingDiff]/maxTf
										upcOldFeatureCount+=1
							
									elif upcOldFeatureCount >= args.upcFeatures and upcFeatureData[trainingDiff][1] >= currentUpcFeatures[minimumUpc]:
										#print "@@#@#@#@#@#@ Replacing %s with %s @#@#@#@#@" % (minimumUpc, trainingDiff)
										del trainingVector[minimumUpc]
										trainingVector.update({trainingDiff:(newFeatures[trainingDiff]/maxTf)})
										currentUpcFeatures[trainingDiff] = newFeatures[trainingDiff]/maxTf
										del currentUpcFeatures[minimumUpc]
							
									elif upcOldFeatureCount >= args.upcFeatures and upcFeatureData[trainingDiff][1] < currentUpcFeatures[minimumUpc]:
										pass
		#print "@@#@#@#@#@#@ All full and %s is not good enough with %s @ %s @#@#@#@#@" %(trainingDiff, minimumUpc, currentUpcFeatures[minimumUpc])

		#							 if upcOldFeatureCount < args.upcFeatures or upcFeatureData[diff][1] > upcOldFeatureMin:
		#								 trainingVector.update({diff:(newFeatures[diff]/maxTf)})
		
		#								 if upcOldFeatureCount >= args.upcFeatures and upcFeatureData[diff][1] > upcOldFeatureMin:
		#									 upcOldFeatureMin = upcFeatureData[diff][1]
		#									 del trainingVector[upcLowestFeature]
		#									 print "lowest %s diff %s" %(upcLowestFeature,diff)
		#									 upcLowestFeature = diff
		#									 upcOldFeatureCount -=1
							
		#								 upcOldFeatureCount+=1

								elif "cpc_" in trainingDiff:
									pass
								
									minimumCpc = min(currentCpcFeatures, key=currentCpcFeatures.get)
								
									if cpcOldFeatureCount < args.cpcFeatures:
										print("@#$@#@# adding %s cpc vector $#@$@#$@" %(trainingDiff))
										trainingVector.update({trainingDiff:(newFeatures[trainingDiff]/maxTf)})
										currentCpcFeatures[trainingDiff] = newFeatures[trainingDiff]/maxTf
										cpcOldFeatureCount+=1
								
									elif cpcOldFeatureCount >= args.cpcFeatures and cpcFeatureData[trainingDiff][1] >= currentCpcFeatures[minimumCpc]:
										#print "@@#@#@#@#@#@ Replacing %s with %s @#@#@#@#@" % (minimumCpc, trainingDiff)
										del trainingVector[minimumCpc]
										trainingVector.update({trainingDiff:(newFeatures[trainingDiff]/maxTf)})
										currentCpcFeatures[trainingDiff] = newFeatures[trainingDiff]/maxTf
										del currentCpcFeatures[minimumCpc]
								
									elif cpcOldFeatureCount >= args.cpcFeatures and cpcFeatureData[trainingDiff][1] < currentCpcFeatures[minimumCpc]:
										pass
										#print "@@#@#@#@#@#@ All full and %s is not good enough with %s @ %s @#@#@#@#@" %(trainingDiff, minimumCpc, currentCpcFeatures[minimumCpc])
								
		#							 if cpcOldFeatureCount < args.cpcFeatures or cpcFeatureData[diff][1] > cpcOldFeatureMin:
		#								 trainingVector.update({diff:(newFeatures[diff]/maxTf)})
		
		#								 if cpcOldFeatureCount >= args.cpcFeatures and cpcFeatureData[diff][1] > cpcOldFeatureMin:
		#									 cpcOldFeatureMin = cpcFeatureData[diff][1]
		#									 del trainingVector[cpcLowestFeature]
		#									 print "lowest %s diff %s" %(cpcLowestFeature,diff)
		#									 cpcLowestFeature = diff
		#									 cpcOldFeatureCount -=1
							
		#								 cpcOldFeatureCount+=1

		# Shock and awe the wrongly guessed label's vectors and envelope
							if euclideanLabelClass != "":
								pass
								with open(euclideanLabelClass+"_vector.txt") as JsonDict:
									wrongVector = json.load(JsonDict)
					
								wrongCommons = list(set(newFeatures.keys()) & set(wrongVector.keys()))
							
								#print "new features is %s, features is %s, and training commons is %s" %(newFeatures,wrongVector.keys(),wrongCommons)
								#print "wrong vectors %s" %wrongVector

								for wrongCommon in wrongCommons:
									pass
									wrongVector[wrongCommon] -= newFeatures[wrongCommon] * (args.wrongWeight/100.0) / maxTf

								vectorWeight = [i ** 2 for i in wrongVector.values()]
								vectorWeight = math.sqrt((100/sum(vectorWeight)))
								for key in wrongVector.keys():
									wrongVector[key] *= vectorWeight

								with open(euclideanLabelClass+"_vector.txt", 'w') as memOut:
									json.dump(wrongVector, memOut)

		# change envelope by a factor
								with open(euclideanLabelClass+"_envelope.txt") as JsonDict:
									wrongEnvelope = json.load(JsonDict)
			
								wrongEnvelope['euclidean'] -= (args.shrinkWrongWeight/10.0)
								wrongEnvelope['cosine'] += (args.shrinkWrongWeight /1000.0)
					
		# scale up shrinkage of envelope to avoid the wrong answer
								if wrongEnvelope['euclidean'] > euclideanHighScore:
									#print "Move FORWARD!!!!!!!"
									wrongEnvelope['euclidean'] = euclideanHighScore
								if wrongEnvelope['cosine'] < cosineHighScore:
									wrongEnvelope['cosine'] = cosineHighScore
				
								with open(euclideanLabelClass+"_envelope.txt", 'w') as memOut:
									json.dump(wrongEnvelope, memOut)

								del wrongVector, wrongCommons, wrongEnvelope

		# scale the vectors to have equidistance of 10
							vectorWeight = [i ** 2 for i in trainingVector.values()]
							vectorWeight = math.sqrt((100/sum(vectorWeight)))
							for key in trainingVector.keys():
								trainingVector[key] *= vectorWeight

								# update lowestFeature for the next row of data
							
									
							del currentSpecFeatures, currentCpcFeatures, currentIpcFeatures, currentUpcFeatures, trainingDiffs
		#del diffs, diff

							with open(traininglabel+"_vector.txt", 'w') as memOut:
								json.dump(trainingVector, memOut)
		#with open(traininglabel+"_envelope.txt", 'w') as memOut:
		#						 json.dump(envelope, memOut)


		# need to write to spec, ipc, upc, cpc files

		############################
		# STOPPED HERE :
		############################

		# Expand the envelope of correct label if it was not captured by the label, with new vectors
		#					 if envelope[
		#					 envelope['euclidean'] -= (args.shrinkCorrectWeight/10.0)
		#					 envelope['cosine'] += (args.shrinkCorrectWeight /1000.0)

		# Recalculate euclidean and cosine score
							rowVector = [0]*(len(trainingVector.values()))

		# try to make it more efficient need to redo trainingCommons
							trainingCommons = list(set(newFeatures.keys()) & set(trainingVector.keys()))
							for trainingCommon in trainingCommons:
								rowVector[trainingVector.keys().index(trainingCommon)] = newFeatures[trainingCommon]/maxTf
		#print "new rowVector %s" %rowVector
		#					 print "new features %s" %newFeatures
		#					 print "training vector keys %s" %trainingVector.keys()

							euclideanScore = numpy.subtract(trainingVector.values(),rowVector)
							euclideanScore = [i ** 2 for i in euclideanScore]
							euclideanScore = sum(euclideanScore)
			
		#print "NEW euclidean score is %s for %s" %(euclideanScore,traininglabel)
			
							cosineScore = numpy.linalg.norm(trainingVector.values())
							cosineScore *= numpy.linalg.norm(rowVector)
							# cosine score of 0 indicates that the vector is perpendicular (1 means parallel), this escapes a div/0 error
							if cosineScore == 0:
								cosineScore = -1
							else:
								cosineScore = numpy.dot(trainingVector.values(),rowVector) / cosineScore
		#print "NEW cosine score is %s for %s" %(cosineScore, traininglabel)

		#print "envelope %s" %envelope

							if envelope['euclidean'] < euclideanScore:
								envelope['euclidean'] = euclideanScore
							if envelope['cosine'] > cosineScore:
								envelope['cosine'] = cosineScore
		#					 print "correct eucl correct cosine"
		#					 print "2 training values %s" %trainingVector.values()
		#					 print "envelope %s" %envelope
							with open(traininglabel+"_envelope.txt", 'w') as memOut:
								json.dump(envelope, memOut)
							del trainingCommons
	   


				specFeatureData["samples"]=specsample
				ipcFeatureData["samples"]=ipcsample
				cpcFeatureData["samples"]=cpcsample
				upcFeatureData["samples"]=upcsample
				with open(traininglabel+"_spec.txt", 'w') as memOut:
					json.dump(specFeatureData, memOut)
				with open(traininglabel+"_ipc.txt", 'w') as memOut:
					json.dump(ipcFeatureData, memOut)
				with open(traininglabel+"_cpc.txt", 'w') as memOut:
					json.dump(cpcFeatureData, memOut)
				with open(traininglabel+"_upc.txt", 'w') as memOut:
					json.dump(upcFeatureData, memOut)

				del learnCsv, specFields, ipcFields, cpcFields, upcFields, rowTerms, learnHeader, fieldname, row, elementTerms, spec, words, word, tf, maxTf, classCol, classString, euclideanHighScore, cosineHighScore, euclideanLabelClass, cosineLabelClass, features, labelVector, rowVector, trainingVector, trainingEnvelope, euclideanScore, cosineScore, newFeatures, specFeatureData, ipcFeatureData, upcFeatureData, cpcFeatureData, specOldFeatureCount, ipcOldFeatureCount, upcOldFeatureCount, cpcOldFeatureCount, oldKeys, featureDataDiffs

		############################################################################
		
		# Brand new vector and envelope being created
			else:
			
				JsonDict = open(traininglabel+"_spec.txt")
				memorySpec = json.load(JsonDict)
				JsonDict = open(traininglabel+"_ipc.txt")
				memoryIPC = json.load(JsonDict)
				JsonDict = open(traininglabel+"_cpc.txt")
				memoryCPC = json.load(JsonDict)
				JsonDict = open(traininglabel+"_upc.txt")
				memoryUPC = json.load(JsonDict)
				JsonDict.close()

				otherLabels = list(labels)
				otherLabels.remove(traininglabel)

				sample = memorySpec.pop("samples")
				numSamples.update({traininglabel+"_spec":sample})
			#	 print "memoryspec before %s" %memorySpec

		# Gather idf for spec features, switch feature count to tf in memorySpec
#				  print(memorySpec)

				## start of new DAW stuff 5/17/2020 #########
				## moved from "array of dicts.py" #########
				
				def setup_otherLabels():
					index = 0
					for otherLabel in otherLabels:
						file_name = otherLabel+"_spec.txt"
						JsonDict = open(file_name)
						print('1468 index: ', index)
						otherDocList[index] = json.load(JsonDict)
						otherDoc = otherDocList[index]
						JsonDict.close()
						otherDocSamplesList[index] = otherDoc['samples']
						print('1473 otherDocSamples: ', otherDocSamplesList[index])
						index = index + 1
	
				#------------------------
				def compute_new_memorySpec():
					print('1478 Start compute_new_memorySpec\n')
					iv = 0
					for term, (count,tf) in memorySpec.items():
						otherSample = 0.0
						df = 0.0
						index2 = 0
						for otherLabel in otherLabels:
							otherDoc = otherDocList[index2]
							otherDocSamples = otherDocSamplesList[index2]
							if term in otherDoc:
								(odf,otf) = otherDoc[term]
								df += odf
							otherSample = otherSample + otherDocSamples
							if (iv==0):
				#								print('1502 otherDocSamples: ', otherDocSamples)
				#								print('1503 otherDoc: ', otherDoc)
				#								print('1509 otherSample: ', otherSample)
								iv += 1
						index2 = index2 + 1
						tfIdf = tf * (math.log(((1.5+otherSample)/(1+df))))
						memorySpec[term] = (tf,tfIdf)
					print('1499 otherSample: ', otherSample)
					print('1500 memorySpec[spec_area]: ',memorySpec['spec_area'])

				#------------------------
				ii = 1
				numberOfLabels = len(otherLabels)
				otherDocSamplesList = [None]*numberOfLabels
				otherDocList = [None]*numberOfLabels
				otherSample = 0.0
				setup_otherLabels()
				compute_new_memorySpec()
				if (ii == 0):
					print('1540 memorySpec after: ', memorySpec)
					ii = ii +1
				## end of new DAW stuff 5/17/2020 #########
				
#------------------------
		# Gather idf for classification features, switch feature count to tf in memoryClass
				sample = memoryIPC.pop("samples")
				numSamples.update({traininglabel+"_ipc":sample})
		#	 print "memoryipc before %s" % memoryIPC
				for term, (count,tf) in memoryIPC.items():
					otherSample = 0.0
					df = 0.0
					for otherLabel in otherLabels:
						JsonDict = open(otherLabel+"_ipc.txt")
						otherDoc = json.load(JsonDict)
						JsonDict.close()
						if term in otherDoc:
							(odf,otf) = otherDoc[term]
							df += odf
						otherSample += otherDoc["samples"]
					tfIdf = tf * (math.log(((1.5+otherSample)/(1+df))))
					memoryIPC[term] = (tf,tfIdf)
		#		 print "term:%s tf:%s tf-idf:%s" % (term,tf,tfIdf)
		#	 print "memoryipc %s" % memoryIPC

				sample = memoryCPC.pop("samples")
				numSamples.update({traininglabel+"_cpc":sample})
		#	 print "memorycpc before %s" % memoryCPC
				for term, (count,tf) in memoryCPC.items():
					otherSample = 0.0
					df = 0.0
					for otherLabel in otherLabels:
						JsonDict = open(otherLabel+"_cpc.txt")
						otherDoc = json.load(JsonDict)
						JsonDict.close()
						if term in otherDoc:
							(odf,otf) = otherDoc[term]
							df += odf
						otherSample += otherDoc["samples"]
		#		 print "df = %f othersample = %f" % (df, otherSample)
					tfIdf = tf * (math.log(((1.5+otherSample)/(1+df))))
					memoryCPC[term] = (tf,tfIdf)
		#		 print "term:%s tf:%s tf-idf:%s" % (term,tf,tfIdf)
		#	 print "memorycpc %s" % memoryCPC

				sample = memoryUPC.pop("samples")
				numSamples.update({traininglabel+"_upc":sample})
		#	 print "memoryupc before %s" % memoryUPC
				for term, (count,tf) in memoryUPC.items():
					otherSample = 0.0
					df = 0.0
					for otherLabel in otherLabels:
						JsonDict = open(otherLabel+"_upc.txt")
						otherDoc = json.load(JsonDict)
						JsonDict.close()
						if term in otherDoc:
							(odf,otf) = otherDoc[term]
							df += odf
						otherSample += otherDoc["samples"]
					tfIdf = tf * (math.log(((1.5+otherSample)/(1+df))))
					memoryUPC[term] = (tf,tfIdf)
		#		 print "term:%s tf:%s tf-idf:%s" % (term,tf,tfIdf)
		#	 print "memoryupc %s" % memoryUPC

		# Select classifying features
				i = 0

		##################
		# Do we want to re-determine what are good features each time? or do you want to go with what you have? Go with what we have, we learn...
		##################

				for term, (tf,tfIdf) in sorted(memorySpec.items(), key=lambda x: x[1][1], reverse=True):
					if i == minSpecFeatures: break
				#count = (count+0.5)/(numSamples[label+"_spec"]+1.5)
				#if not term in scoreCardSpec:
				#	scoreCardSpec.update({term:{label:tf}})
				#else:
				#	scoreCardSpec[term].update({label:tf})
					trainingVector.update({term:tf})
					i += 1

				i = 0
				for term, (tf,tfIdf) in sorted(memoryIPC.items(), key=lambda x: x[1][1], reverse=True):
					if i == minIpcFeatures: break
				#count = (count+0.5)/(numSamples[label+"_ipc"]+1.5)
				#		 if not term in scoreCardIPC:
				#	 scoreCardIPC.update({term:{label:tf}})
				#else:
				#	 scoreCardIPC[term].update({label:tf})
					trainingVector.update({term:tf})
					i += 1

				i = 0
				for term, (tf,tfIdf) in sorted(memoryCPC.items(), key=lambda x: x[1][1], reverse=True):
					if i == minCpcFeatures: break
				#count = (count+0.5)/(numSamples[label+"_cpc"]+1.5)
				#if not term in scoreCardCPC:
				#	 scoreCardCPC.update({term:{label:tf}})
				#else:
				#	 scoreCardCPC[term].update({label:tf})
					trainingVector.update({term:tf})
					i += 1

				i = 0
				for term, (tf,tfIdf) in sorted(memoryUPC.items(), key=lambda x: x[1][1], reverse=True):
					if i == minUpcFeatures: break
				#count = (count+0.5)/(numSamples[label+"_upc"]+1.5)
				#if not term in scoreCardUPC:
				#	 scoreCardUPC.update({term:{label:tf}})
				#else:
				#	 scoreCardUPC[term].update({label:tf})
					trainingVector.update({term:tf})
					i += 1

		# Normalize the euclidean distance of training vectors to 10 from origin, note that euclidean score is the squared of actual euclidean distance
				vectorWeight = [i ** 2 for i in trainingVector.values()]
				vectorWeight = math.sqrt((100/sum(vectorWeight)))
				for key in trainingVector.keys():
					trainingVector[key] *= vectorWeight
		#		 trainingVector.update({term:tf})
		#print "training vector of %s is %s" %(label,trainingVector)

		# determine the envelope of label vector
				specFields = []
				ipcFields = []
				cpcFields = []
				upcFields = []
		#	 specTf = Counter()
		#	 ipcTf = Counter()
		#	 upcTf = Counter()
		#	 cpcTf = Counter()
		#	 numSpecRows = 0
			#Actual rows with PC rows is numI/U/CpcRows + numSpecRows
		#	 numIpcRows = 0
		#	 numCpcRows = 0
		#	 numUpcRows = 0

				features = []
				labelVector = []
				rowVector = []
				trainingEnvelope = {}

				features = list(trainingVector.keys())
				labelVector = list(trainingVector.values())
				euclideanBest = 0.0
				cosineBest = 1.0

				with open(traininglabel+".csv", 'rU') as train_FH:
					#print("1576 Training %s" %(traininglabel))
					trainCsv = csv.DictReader (train_FH)
					for fieldname in trainCsv.fieldnames:
						if re.search(SPEC_HEADERS,fieldname, re.I):
							specFields.append(fieldname)
						elif re.search(IPC_HEADERS,fieldname, re.I):
							ipcFields.append(fieldname)
						elif re.search(CPC_HEADERS,fieldname, re.I):
							cpcFields.append(fieldname)
						elif re.search(UPC_HEADERS,fieldname, re.I):
							upcFields.append(fieldname)

					for row in trainCsv:
						rowTerms=[]
						rowVector = [0.0]*(len(labelVector))
					#numSpecRows += 1
					#Gather specs terms
						for spec in specFields:
							words = [wnl.lemmatize(w) for w in nltk.wordpunct_tokenize(row[spec])]
							for word in words:
								word = word.lower()
								if word not in stopwords:
									if word.isalpha():
										if "spec_"+word in features:
										#print "found %s in features at label %s" %(word,label)
											rowVector[features.index("spec_"+word)] +=1.0
								
									#rowTerms.append(word)
					
						#specTf = Counter(rowTerms)
						
		#check if no specs
		#if rowTerms:
		#					 maxTf = specTf.most_common(1)[0][1]
		#				 else:
		#					 maxTf = 1

		#normalize for length of specs
						maxTf = max(rowVector)
					#print "max tf %s" %maxTf
						if maxTf != 0:
							for i in features:
								if "spec_" in i:
									rowVector[features.index(i)] /=maxTf
				#print "features %s" %features
		#print "row vector %s" % rowVector
		#			 for (term,count) in specTf.items():
		#				 count = count/(1.0*maxTf)
		#				 specTf[term]=0
		#				 specTf.update({term:count})

		#Gather classifications
						for classCol in upcFields:
							classString = row[classCol]
							if classString=="":
							#numUpcRows -= 1
								break
							rowTerms.extend(inno_clean_pc(classString,"upc"))
						if rowTerms:
						#print rowTerms
							for rowTerm in rowTerms:
								if rowTerm in features:
									rowVector[features.index(rowTerm)] +=1.0
							maxTf = max(rowVector)
							if maxTf != 0:
								for i in features:
									if "upc_" in i:
										rowVector[features.index(i)] /=maxTf
		#				 print "row vector UPC %s" %rowVector
		#upcTf = Counter(rowTerms)
		#maxTf = upcTf.most_common(1)[0][1]
		#for (term,count) in upcTf.items():
		#count = count/(1.0*maxTf)
		#upcTf[term]=0
		#upcTf.update({term:count})
						rowTerms=[]

						for classCol in cpcFields:
							classString = row[classCol]
							if classString=="":
							#numCpcRows -= 1
								break
							rowTerms = inno_clean_pc(classString,"cpc")
						if rowTerms:
							for rowTerm in rowTerms:
								if rowTerm in features:
									rowVector[features.index(rowTerm)] +=1.0
							maxTf = max(rowVector)
							if maxTf != 0:
								for i in features:
									if "cpc_" in i:
										rowVector[features.index(i)] /=maxTf
					
		#				 cpcTf = Counter(rowTerms)
		#				 maxTf = cpcTf.most_common(1)[0][1]
		#				 for (term,count) in cpcTf.items():
		#					 count = count/(1.0*maxTf)
		#					 cpcTf[term]=0
		#					 cpcTf.update({term:count})
						rowTerms=[]

						for classCol in ipcFields:
							classString = row[classCol]
							if classString=="":
							#					 numIpcRows -= 1
								break
							rowTerms = inno_clean_pc(classString,"ipc")
						if rowTerms:
							for rowTerm in rowTerms:
								if rowTerm in features:
									rowVector[features.index(rowTerm)] +=1.0
							maxTf = max(rowVector)
							if maxTf != 0:
								for i in features:
									if "ipc_" in i:
										rowVector[features.index(i)] /=maxTf

		#				 ipcTf = Counter(rowTerms)
		#				 maxTf = ipcTf.most_common(1)[0][1]
		#				 for (term,count) in ipcTf.items():
		#					 count = count/(1.0*maxTf)
		#					 ipcTf[term]=0
		#					 ipcTf.update({term:count})
						rowTerms=[]
					#print "row vector %s" %rowVector

		# calculate euclidean scores
						euclideanScore = numpy.subtract(labelVector,rowVector)
						euclideanScore = [i ** 2 for i in euclideanScore]
						euclideanScore = sum(euclideanScore)
				
						if euclideanScore > euclideanBest: euclideanBest = euclideanScore

		# calculate cosine scores
						cosineScore = numpy.linalg.norm(labelVector)
						cosineScore *= numpy.linalg.norm(rowVector)
						if cosineScore == 0:
							cosineScore = 1
						else:
							cosineScore = numpy.dot(labelVector,rowVector) / cosineScore
				
						if cosineScore < cosineBest: cosineBest = cosineScore

				trainingEnvelope.update({'euclidean':euclideanBest})
				trainingEnvelope.update({'cosine':cosineBest})


						#			 print "spec tf %s" %specTf
		#print "ipc tf %s" %ipcTf

		#	 print "spec tf %s" %specTf
				with open(traininglabel+"_vector.txt", 'w') as memOut:
					json.dump(trainingVector, memOut)
		
				with open(traininglabel+"_envelope.txt", 'w') as memOut:
					json.dump(trainingEnvelope,memOut)

				trainingVector.clear()
				del otherLabels, otherSample, i, sample, vectorWeight, train_FH, trainCsv, specFields, ipcFields, cpcFields, upcFields, fieldname, row, rowTerms, spec, words, word, tf, maxTf, term, count, classCol, classString, features, labelVector, rowVector, rowTerm, euclideanScore, cosineScore, trainingEnvelope, memorySpec, memoryIPC, memoryCPC, memoryUPC

			del traininglabel

		print("1738 end training()")
		return args, labels, headers, wnl, stopwords

#print "scorecard IPC %s" % scoreCardIPC
#print "scorecard spec %s" % scoreCardSpec
#print "scorecard UPC %s" % scoreCardUPC
#print "scorecard CPC %s" % scoreCardCPC

##-------------------------------------------------------------------------------------------------------------
def classify_results(args, labels):
# documenting classification results
	print("1650 start classify_results()")
	temps=[]
	with open(args.result,'w') as class_FH:
		classData = csv.writer(class_FH)
		for label in labels:
			temps.append("Features")
			temps.append(label)
	#	 temps.insert(0,"Features")
			classData.writerow(temps)
			JsonDict = open(label+"_vector.txt")
			trainingVector = json.load(JsonDict)

			for term, tf in trainingVector.items():
				del temps[:]
				temps.append(term)
				temps.append(tf)
				classData.writerow(temps)
		
			del temps[:]
			classData.writerow(temps)
	del classData, temps, trainingVector
	print("1671 end classify_results()")
	return args, labels


# Classify target file
##-------------------------------------------------------------------------------------------------------------
def classify_target_file(args, headers, labels, wnl, stopwords):
	print("1677 start classify_target_file()")
	SPEC_HEADERS = headers[0]
	IPC_HEADERS = headers[1]
	CPC_HEADERS = headers[2]
	UPC_HEADERS = headers[3]
	
	targetFile = args.targetFile
	if targetFile:
		#print("1743 Classifying %s" % (targetFile))
		#print "training vector is %s" %trainingVector
		#	 print "label is %s" % labels
	# Loop for each row in targetFile.csv
	#	 print args.targetFile
		resultFile = re.sub(r'(\w?).csv',r'\1_result.csv',targetFile)
		resultBuffer = []
		resultHeader = []
		euclideanScore = []
		cosineScore = []
		with open (resultFile, 'w') as result_FH:
			resultData = csv.writer(result_FH)
			with open (targetFile, 'rU') as target_FH:
				specFields=[]
				ipcFields=[]
				cpcFields=[]
				upcFields=[]
				rowTerms=[]
				targetCsv = csv.DictReader (target_FH)
	#Segment search terms, text and patent class
				for fieldname in targetCsv.fieldnames:
					if re.search(SPEC_HEADERS,fieldname, re.I):
						specFields.append(fieldname)
					elif re.search(IPC_HEADERS,fieldname, re.I):
						ipcFields.append(fieldname)
					elif re.search(CPC_HEADERS,fieldname, re.I):
						cpcFields.append(fieldname)
					elif re.search(UPC_HEADERS,fieldname, re.I):
						upcFields.append(fieldname)
					resultHeader.append(fieldname)
				resultBuffer = resultHeader[:]
				for label in labels:
					resultBuffer.append(label+"_euclidean")
					resultBuffer.append(label+"_cosine")
				#			 resultBuffer.extend(labels)
				resultBuffer.extend(["Classification_euclidean","Classification_cosine"])
				resultData.writerow(resultBuffer)
				resultBuffer=[]
				for row in targetCsv:
					#specTerms=[]
					#ipcTerms=[]
					#upcTerms=[]
					#cpcTerms=[]
					elementTerms=[]
					#print resultHeader
	#Populate patent data to result file
					for header in resultHeader:
						resultBuffer.append(row[header])
					#print "result buffer %s" %resultBuffer
					# delete?
					rowTerms=[]
	#Reset scores for each label to equal probability
	#				 resultScore = resultScore.fromkeys(labels, (1.0/len(labels)))

	#Gather specs terms
					for spec in specFields:
						words = [wnl.lemmatize(w) for w in nltk.wordpunct_tokenize(row[spec])]
						for word in words:
							word = word.lower()
							if word not in stopwords:
								if word.isalpha():
									word="spec_"+word
									rowTerms.append(word)
									
					tf = Counter(rowTerms)
	#check if no specs
					if rowTerms:
						maxTf = tf.most_common(1)[0][1] * 1.0
					else:
						maxTf = 1.0
				
					#print "max tf %s" %maxTf

					#if not term in scoreCardSpec:
						#scoreCardSpec.update({term:{label:tf}})
				
			#what if no specs?
	#Remove duplicated spec terms
	#				 print "target spec terms %s" %rowTerms
	#				 specTerms = list(set(specTerms))
	#				 resultScore = calculate_score(scoreCardSpec,numSamples,specTerms,resultScore,"spec")

	#Gather classifications
					for classCol in upcFields:
						classString = row[classCol]
						if classString=="":
							break
						elementTerms = inno_clean_pc(classString,"upc")
						rowTerms.extend(elementTerms)
	#				 upcTerms = list(set(upcTerms))
	#				resultScore = calculate_score(scoreCardUPC,numSamples,upcTerms,resultScore,"upc")
					elementTerms=[]

					for classCol in ipcFields:
						classString = row[classCol]
						if classString=="":
							break
						elementTerms = inno_clean_pc(classString,"ipc")
						rowTerms.extend(elementTerms)
	#				 ipcTerms = list(set(ipcTerms))
	#				 resultScore = calculate_score(scoreCardIPC,numSamples,ipcTerms,resultScore,"ipc")
					elementTerms=[]

					for classCol in cpcFields:
						classString = row[classCol]
						if classString=="":
							break
						elementTerms = inno_clean_pc(classString,"cpc")
						rowTerms.extend(elementTerms)
	#				 cpcTerms = list(set(cpcTerms))
	#				 resultScore = calculate_score(scoreCardCPC,numSamples,cpcTerms,resultScore,"cpc")
					elementTerms=[]

	#Evaluate the labels one at a time, Record in result buffer
	#highScore = float("inf")
					euclideanHighScore = float("inf")
					cosineHighScore = -1.0
						#labelclass = ""
					euclideanLabelClass = ""
					cosineLabelClass = ""
				
					#print "row terms are %s" %rowTerms
					for label in labels:
						features =[]
						labelVector = []
						rowVector = []

	#					 resultBuffer.append(resultScore[label])

						with open(label+"_vector.txt") as JsonDict:
							trainingVector = json.load(JsonDict)
						with open(label+"_envelope.txt") as JsonDict:
							trainingEnvelope = json.load(JsonDict)

	#print "training vector for label %s is %s" %(label,trainingVector)
						features = list(trainingVector.keys())
						#print "features are %s" %features
						labelVector = list(trainingVector.values())
						#print "label vectors are %s" %labelVector
						rowVector = [0.0]*(len(labelVector))
						#print "row vector %s" %rowVector

						for rowTerm in rowTerms:
							if rowTerm in features:
								#print "found %s in features" %rowTerm
								#print "here %s" %features.index(rowTerm)
								rowVector[features.index(rowTerm)] +=1
						rowVector = [i/maxTf for i in rowVector]
					
						euclideanScore = numpy.subtract(labelVector,rowVector)
						euclideanScore = [i ** 2 for i in euclideanScore]
						#print "euclidean score %s" %euclideanScore
						euclideanScore = sum(euclideanScore)
						#print "euclidean score %s" %euclideanScore
						resultBuffer.append(euclideanScore)
					
						cosineScore = numpy.linalg.norm(labelVector)
						#print "cosine score %s" %cosineScore
						cosineScore *= numpy.linalg.norm(rowVector)
						#print "cosine score %s" %cosineScore
						if cosineScore == 0:
							cosineScore = -1
						else:
							cosineScore = numpy.dot(labelVector,rowVector) / cosineScore
						#print "cosine score %s" %cosineScore
						resultBuffer.append(cosineScore)

	##################################
	# Think about doing mutually inclusive labels
	##################################

	# Determine label for row
	#print "row betors %s" %rowVector
						if euclideanScore<=trainingEnvelope['euclidean']:
							if euclideanLabelClass == "":
								euclideanLabelClass=label
								euclideanHighScore =euclideanScore
							else:
								if euclideanScore<euclideanHighScore:
									euclideanLabelClass=label
									euclideanHighScore =euclideanScore
								
						if cosineScore>=trainingEnvelope['cosine']:
							if cosineLabelClass == "":
								cosineLabelClass=label
								cosineHighScore =cosineScore
							else:
								if cosineScore>cosineHighScore:
									cosineLabelClass=label
									cosineHighScore =cosineScore

	#if euclideanScore<highScore:
	#						 labelclass=label
	#						 highScore=euclideanScore
	#					 elif euclideanScore==highScore:
	#						 labelclass=""
						
					resultBuffer.extend([euclideanLabelClass,cosineLabelClass])

					resultData.writerow(resultBuffer)
					resultBuffer=[]
					rowTerms=[]
	#		  del resultFile, resultBuffer, resultHeader, specFields, ipcFields, upcFields,
	#		  cpcFields, spec, fieldname, row, targetCsv, resultData, words, word, euclideanScore, '''label,'''
	#		  elementTerms, rowTerms,''' JsonDict,''' trainingVector, labelVector, features, rowTerm, cosineScore, trainingEnvelope, euclideanHighScore, cosineHighScore, maxTf
		del resultFile, resultBuffer, resultHeader, specFields, ipcFields, upcFields,
		cpcFields, spec, fieldname, row, targetCsv, resultData, words, word, euclideanScore,
		elementTerms, rowTerms,trainingVector, labelVector, features, rowTerm, cosineScore, trainingEnvelope,
		euclideanHighScore, cosineHighScore, maxTf
	print("1894 stop classify_target_file()")

##-------------------------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------------------------

	##### TO DO #####
	# 1. clean up ",\r\n in csv files
	# 2. warn if only one file is being trained, no others to compare and would be meaningless
	# 3. log learnings and times
	##### END TO DO #####
	