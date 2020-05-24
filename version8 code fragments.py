#------------------------
def setup_otherDocNames():
    i = 0
    for otherDocName in otherLabels:
        otherDocName = 'otherDoc' + '_' + str(i)
        print('1466 otherDocName', otherDocName, flush=True)
        i = i + 1


#------------------------
def setup_otherLabels():
    index = 0
    for otherLabel in otherLabels:
        file_name = otherLabel+"_spec.txt"
        print('1476 file_name:', file_name, flush=True)
        JsonDict = open(file_name)
        otherDocName = 'otherDoc' + '_' + str(index)
        otherDocName = json.load(JsonDict)
        JsonDict.close()
        otherDocNameSamples = float(otherDocName['samples'])
        index = index + 1
    
#------------------------
def compute_new_memorySpec():
    print('1500 Start compute_new_memorySpec', flush=True)
    items_count = 0             
    for term, (count,tf) in memorySpec.items():
        otherSample = 0.0
        df = 0.0
        index2 = 0
        for otherLabel in otherLabels:
            otherDocName = 'otherDoc' + '_' + str(index2)
            if term in otherDocName:
                (odf,otf) = otherDocName[term]
                df += odf
            otherSample = otherSample + float(otherDocNameSamples)
        index2 = index2 + 1
        tfIdf = tf * (math.log(((1.5+otherSample)/(1+df))))
        items_count = items_count + 1
        memorySpec[term] = (tf,tfIdf)
    if (items_count == 3520):
        print('1519 term, (tf,tfIdf): ',term, (tf,tfIdf))
    print('1535 End compute_new_memorySpec', flush=True)            

#------------------------
ii = 1
numberListElements = len(otherLabels)
otherDocNameSamples = 0.0
setup_otherDocNames()
setup_otherLabels()
compute_new_memorySpec()

