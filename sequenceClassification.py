import re
import csv
import json
import pickle
from pathlib import Path
from termSemanticType import *
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix

stopWords = set(stopwords.words('english'))
umls_cache_dir = Path('cache/')
umls_cache_file = Path('semanticType.pb')

if not umls_cache_dir.is_dir():
    print("output dir does not exists, creating a new one.")
    umls_cache_dir.mkdir()
umls_cache_file = umls_cache_dir.joinpath(umls_cache_file)


class ClassifiedSequence(object):
    def __init__(self, sentence, recordObj=None):
        self.sentence = sentence
        self.record = []
        self.record.append(recordObj)
        self.classified = 0
        self.attributeKey = 0
        self.attributeValue = 0
        self.truthClass = 0
        self.attrKeys = {}
        self.predictedClass = 0

    def __hash__(self):
        return hash((self.sentence))

    def __eq__(self, other):
        if not isinstance(other, type(self)): 
            return NotImplemented

        return self.sentence == other.sentence
    
    def __str__(self):
        string = '%s;;%f;;%f;;%f;;%f;;%s;;%f;;%s' % (self.sentence, self.classified, self.attributeKey,  self.attributeValue, self.truthClass, str(self.attrKeys), self.predictedClass, str(self.record))
        return string

    @staticmethod
    def header():
        return 'sentence;; classified;; attributeKey;;  attributeValue;; truthClass;; predictedClass'

class SequenceClassification:

    def __init__(self):

        self.updateTermSemanticType = False
        self.termSemanticTypeObj = TermSemanticType()
        self.drug = ['Organic Chemical · Pharmacologic Substance ']
        self.semanticTypes = {}
        self.extractorFunctionRegex = ['name', 'age', 'duration', 'frequency', 'symptoms']

        print("Cache file path:" + str(umls_cache_file.absolute()))
        if umls_cache_file.is_file():
            with open(umls_cache_file, 'rb') as file:
                self.semanticTypes = pickle.load(file)
        else:
            print("Cache file not found. Will create a new one when enough data has been collected.")

    def printCacheStats(self):
        bigramTokens = 0
        unigramTokens = 0
        for key,value in self.semanticTypes.items():
            if " " in key:
                bigramTokens += 1
            else:
                unigramTokens += 1
        print("Total token in cache:", len(self.semanticTypes.keys()))
        print("Unigram tokens in cache:", unigramTokens)
        print("bigram tokens in cache:", bigramTokens)

    def readFile(self, filePath):

        data = []

        with open(filePath) as f:
            content = f.readlines()

        for row in content:
            row = row.replace('\n', '')
            rowArray = row.split(';;')
            data.append(rowArray)

        return data

    def writeCSV(self, filePath, data):

        # print('data: ', data)

        with open(filePath, mode='w', encoding='UTF8', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
    
    def writeOutputForClassifiedSequence(self, data, output_filepath = None):
        if output_filepath is None:
            print(str(ClassifiedSequence.header()).replace(";;", "\t"))
            for c_seq in data:
                print(c_seq.replace(";;", "\t"))
        else:
            print("Writing output file at:", Path(output_filepath).absolute())
            with open(output_filepath, 'w') as ofp:
                ofp.write(str(ClassifiedSequence.header()))
                ofp.write("\n")
                for c_seq in data:
                    ofp.write(str(c_seq))
                    ofp.write("\n")
        
    def getBigramTokens(self, tokens):

        tokensWithoutStopWords = [t for t in tokens if t not in stopWords]
        bigrams = [b for b in zip(tokensWithoutStopWords[:-1], tokensWithoutStopWords[1:])]

        return bigrams

    def isTypeInTokenSemantics(self, token, type):
        typeFound = False
        # If the token is not present in the file then we should call UMLS

        if len(token.strip()) < 1:
            return False

        if token not in self.semanticTypes:
            tokenSemantics = self.termSemanticTypeObj.getTermSemanticTypes(token)
            self.semanticTypes[token] = [t.decode('utf-8') for t in tokenSemantics]
            self.updateTermSemanticType = True

        for semType in self.semanticTypes[token]:
            semType = semType.strip().lower()
            if isinstance(semType, str) and type == 'drug' and semType in self.drug:
                return True
            if isinstance(semType, str) and semType == type:
                return True
        return False

    def getActualAndPredictedValue(self, record):
        result = []
        sentence = record[0].lower().strip()

        sentence = sentence.replace('[cls]', '').replace('[sep] ', '').replace('[sep]', '').strip()

        orignalAttrArray = []
        predictedAttrArray = []
        similarity = 0.0
        pattern = ""

        testSentence = False

        #print('record: ', record)

        # if sentence == 'she has a fever': # == 'due to which also having problem with breathing':
        #     testSentence = True

        
        if len(record)>1:
            orignalAttr = record[1]
            if testSentence:
                print('orignalAttr: ', orignalAttr)
                
            orignalAttrArray = orignalAttr.lower().split(',')
        if len(record) > 2:
            predictedAttr = record[2]
            predictedAttrArray = [x.split(':')[0].strip() for x in predictedAttr.lower().split(',') if x]
            if testSentence:
                print('predictedAttrArray: ', predictedAttrArray)
        if len(record) > 3:
            similarity = record[3]
        if len(record) > 5:
            pattern = record[5].strip()

        _predictedAttributeMap = {}

        tokens = sentence.split(' ')  # tokenization via space
        # print('sentence: ', sentence)
        # print('tokens: ', tokens)



        bigramTokens = self.getBigramTokens(tokens)

        # for token in bigramTokens:
        #     tokenString = " ".join(list(token))
        #     if self.isTypeInTokenSemantics(tokenString, ""):
        #         predictedAttributeMap['-'] = tokenString

        # for token in tokens:
        #     if token in stopWords:
        #         continue
        #     if self.isTypeInTokenSemantics(token, ""):
        #         predictedAttributeMap['-'] = token

        for pred_attr in predictedAttrArray:
            # print('pred_attr: ', pred_attr)
            if pred_attr not in self.extractorFunctionRegex:
                # print('now goint to call urml')
                # tokens = sentence.split(' ')  # tokenization via space
                # print('sentence: ', sentence)
                # print('tokens: ', tokens)

                # bigramTokens = self.getBigramTokens(tokens)

                for token in bigramTokens:
                    tokenString = " ".join(list(token))

                    tokenString = tokenString.replace('?', '').replace('.','')

                    if self.isTypeInTokenSemantics(tokenString, pred_attr):
                        if pred_attr not in _predictedAttributeMap:
                            _predictedAttributeMap[pred_attr] = []
                        _predictedAttributeMap[pred_attr].append(tokenString)
                        
                if testSentence:
                    print('_predictedAttributeMap: ', _predictedAttributeMap)
                
                for token in tokens:

                    token = token.replace('?', '').replace('.','')

                    if testSentence:
                        print("token: ", token)
                    if token in stopWords:
                        continue
                    if self.isTypeInTokenSemantics(token, pred_attr):
                        if pred_attr not in _predictedAttributeMap:
                            _predictedAttributeMap[pred_attr] = []
                        _predictedAttributeMap[pred_attr].append(token)

                if testSentence:
                    print('_predictedAttributeMap: ', _predictedAttributeMap)
   
            else:
                try:
                    matches = re.search(pattern, sentence)
                    if pred_attr not in _predictedAttributeMap:
                        _predictedAttributeMap[pred_attr] = []
                        # if sentence.__contains__("and what is her age?"):
                        #     print(_predictedAttributeMap[pred_attr])
                        #     print("_predictedAttributeMap:",_predictedAttributeMap, ", pred_attr:", pred_attr)

                    if pred_attr == 'name':
                        _predictedAttributeMap[pred_attr].append(matches.group('Name'))
                    elif pred_attr == 'age':
                        _predictedAttributeMap[pred_attr].append(matches.group('Age'))
                        # if sentence.__contains__("and what is her age?"):
                        #     print(_predictedAttributeMap[pred_attr])

                        # if (sentence == "and what is her age? 6 years" and matches.group('Age') == 'and '):
                        #     print('record with target sentence: ', record)
                        

                        # if(pattern == '(old|age)(.*)?\\? (he is|she is|shes)?(?P<Age>.*)(years|month)?(.*)?' and matches.group('Age') != ''):
                        #     print('sentence: ', sentence, ' value: ', matches.group('Age'))

                    elif pred_attr == 'duration':
                        _predictedAttributeMap[pred_attr].append(matches.group('Duration'))
                    elif pred_attr == 'frequency':
                        _predictedAttributeMap[pred_attr].append(matches.group('Frequency'))
                    elif pred_attr == 'symptoms':
                        _predictedAttributeMap[pred_attr].append(matches.group('Symptoms'))
                    else:
                        print("No extraction method found for:", pred_attr)
                except:
                    skip = False
                    # print(":")

        
        orignalAttrMap = {}
        for t in orignalAttrArray:
            _a = t.split(':')
            if len(_a) > 0 and _a[0] != "":
                _key = _a[0].lower().strip()
                orignalAttrMap[_key] = []
                if len(_a) > 1 and _a[1] != "":
                    orignalAttrMap[_key].append(_a[1].lower())

        attr = "notequal"
        setEqual = True
            

        if len(orignalAttrMap.keys()) < 1 and len(_predictedAttributeMap.keys()) > 0:
            for key, value in _predictedAttributeMap.items():
                if testSentence and key == 'frequency':
                    print('key: ', key, ' value: ', value)
                    exit()
                if value != '':
                    setEqual = False
                    attr = 'notequal'
                    break

        for oriName in orignalAttrMap.keys():
            if oriName not in _predictedAttributeMap:
                setEqual = False
            else:
                if orignalAttrMap[oriName] == _predictedAttributeMap[oriName]:
                    if attr == "notequal":
                        attr = "partial"
                else:
                    setEqual = False

        if setEqual:
            attr = "equal"

        if attr == "equal":
            correctIdentifed = '1'
            orignalLable = '1'
        elif attr == "partial":
            correctIdentifed = '~'
            orignalLable = '~'
        else:
            correctIdentifed = '0'
            orignalLable = '0'

        result.append(sentence)
        result.append(orignalAttrMap)
        result.append(_predictedAttributeMap)
        result.append(similarity)
        result.append(correctIdentifed)
        result.append(orignalLable)

        if testSentence:
            print(result)



        # print('result: ', result)

        return result

    def getPatternMatchedSentence(self, data):

        result = []
        correctIdentifiedRecord = []
        for index, record in enumerate(data):
            # if (index % 1000 == 0):
            #     print('index: ', index)
            if index != 0 and index % 50 == 0:
                if (self.updateTermSemanticType):
                    with open(umls_cache_file, "wb+") as fp:
                        pickle.dump(self.semanticTypes, fp)
                        self.updateTermSemanticType = False
            matches = self.getActualAndPredictedValue(record)
            if len(matches) > 0:
                result.append(matches)
        if (self.updateTermSemanticType):
            with open(umls_cache_file, "wb+") as fp:
                pickle.dump(self.semanticTypes, fp)

        return result


    def getTestSentences(self):

        originalSequences = obj.readFile("Data/TestData/transcript_test_label_musarrat_2.4.csv")
        originalSequences_set = {}
        for record in originalSequences:

            record_seq = record[0].strip().lower()
            if(record_seq not in originalSequences_set.keys()):
                # record_seq = record_seq.lower()
                originalSequences_set[record_seq] = ClassifiedSequence(record_seq, record)

                if len(record) > 1 and record[1] != '':

                    # print('record in lenght greater 1: ', record)

                    originalSequences_set[record_seq].truthClass =  1
                    attributesList = record[1].split(",");
                    for attrib in attributesList:
                        if(attrib==""):
                            continue
                        attrib = attrib.strip();
                        items = attrib.split(":")
                        key = items[0].strip().lower()
                        if(items[0] not in originalSequences_set[record_seq].attrKeys.keys()):
                            originalSequences_set[record_seq].attrKeys[key] = []

                        if(len(items)>1):
                            originalSequences_set[record_seq].attrKeys[key].append(items[1].strip().lower())
                        else:
                            print("record: ", record)
                            print(attributesList,  attrib, "=", items)

                    # originalSequences_set[record_seq].attrKeys = record[1]
                else:
                    originalSequences_set[record_seq].truthClass =  0

        return originalSequences_set
        
    
    def getTruePossitiveSetnences(self, data):


        TPSentences = {}
        wrongSentences = {}
        falsePositiveSentences = {}  # Ori label is empty but we have classified it

        originalSequences_set = self.getTestSentences()
        # truthClasses = [x.truthClass for x in originalSequences_set.values()]
        # print(truthClasses)

        testSentence = False
        output_record = {}
        testcount = 0
        seq = ""

        for index, record in enumerate(data):

            recordSeq = record[0].strip().lower()

            # print('recordSeq: ', recordSeq)
            # recordSeq = recordSeq.replace('[cls]', '').replace('[sep] ', '').replace('[sep]', '').strip()
            
            # if recordSeq == "did she catch a cold? yes":
            #     testSentence = True
            # else:
            #     testSentence = False

            # if testSentence:
            #     print('record: ', record)
            # else:
            #     continue



            originalSequences_set[recordSeq].classified = 1

            oriAttrNameArray = list(record[1].keys())
            predictedAttrNameArray = list(record[2].keys())

            if testSentence:
                print('oriAttrNameArray: ', oriAttrNameArray, ' predictedAttrNameArray: ', predictedAttrNameArray)
            attrKey = "notequal"
            attrValue = "notequal"
            setEqual = True

            if len(oriAttrNameArray) < 1 and len(predictedAttrNameArray) < 1:
                originalSequences_set[recordSeq].predictedClass = 0
            # elif len(oriAttrNameArray) < 1 and len(predictedAttrNameArray) > 0:
            #     originalSequences_set[recordSeq].predictedClass = 0
            else:
                probableKeyFound = False
                probableValueFound = False
                # this loop and if is only to check attribute key
                if len(oriAttrNameArray) < 1:
                    setEqual = False
                    probableKeyFound = True

                for oriName in oriAttrNameArray:
                    if oriName not in predictedAttrNameArray:
                        setEqual = False
                    else:
                        if attrKey == "notequal":
                            attrKey = "partial"
                        
                if setEqual:
                    attrKey = "equal"

                # if key is equal or partical and there is any extrated value exists
                
                if testSentence:
                    print('attrKey: ', attrKey, ' attrValue: ', attrValue, " list(record[2].keys()):", list(record[2].keys()))

                setEqual = True
                for i, extKey in enumerate(list(record[2].keys())):
                    # print('i:',str(i))
                    if testSentence:
                        print('extKey: ', extKey)

                    if len(record[2][extKey]) > 0:
                        # setEqual  = False
                        probableValueFound = True

                    extKey = extKey.strip()
                    # print('extKey2: ', extKey)

                    # print('extKey: ', extKey, ' len(record[2][extKey]): ', len(record[2][extKey]), 'originalSequences_set[recordSeq].attrKeys.keys(): ', originalSequences_set[recordSeq].attrKeys.keys())
                    # print('extKey: ', extKey, ' record: ', record)
                    if len(record[2][extKey]) > 0 and extKey in originalSequences_set[recordSeq].attrKeys.keys():
                    # if extKey in originalSequences_set[recordSeq].attrKeys.keys():
                        for attr_value in record[2][extKey]:
                            # print('attr_value: ', attr_value)
                            attr_value = attr_value.strip().lower()
                            try:
                                if attr_value not in originalSequences_set[recordSeq].attrKeys[extKey]:
                                    attrValue = 'partial'
                                    # if(extKey == 'age'):
                                    # print('record: ', record)
                                    # print("value not in list:",attr_value)
                                    # print("extKey:", extKey)
                                    # print("originalSequences_set[recordSeq].attrKeys[extKey]:",originalSequences_set[recordSeq].attrKeys[extKey])
                                    
                                    # exit()
                                # else:
                                #     print("VALUE FOUND")
                                #     print('attr_value: ', attrValue)
                                    
                            except:
                                print('record: ', record[2])
                                print('extKey: ', extKey)
                                print(originalSequences_set[recordSeq].attrKeys)
                                exit()

                        # if record[2][extKey] in originalSequences_set[recordSeq].attrKeys[extKey]:
                        #     print("value found")
                        # else:
                        #     print("value not in list")

                        # attrValue = 'partial'
                        # print(originalSequences_set[recordSeq].attrKeys[record[2][extKey]],'--',record[2][extKey])
                        
                    else:
                        setEqual = False
                
                if setEqual:
                    attrValue = 'equal'
                    # print('setEqual: ', setEqual)
                    
                
                if testSentence:
                    print('attrKey: ', attrKey, ' attrValue: ', attrValue)


                # if (attrKey == "equal" or attrKey == "partial"):
                #     originalSequences_set[recordSeq].attributeKey = 1
                # else:
                #     if len(oriAttrNameArray) < 1:
                #         originalSequences_set[recordSeq].attributeKey = 2
                #     else:
                #         originalSequences_set[recordSeq].attributeKey = 0

                if probableKeyFound and probableValueFound:
                    originalSequences_set[recordSeq].predictedClass = 1
                    continue

                test = False

                if attrValue == 'equal' and attrKey == "equal":
                    originalSequences_set[recordSeq].predictedClass = 1
                    test = True
                elif (attrValue == 'partial' and attrKey == "equal") or (attrValue == 'equal' and attrKey == "partial"):
                    if(originalSequences_set[recordSeq].predictedClass < 0.5):
                        originalSequences_set[recordSeq].predictedClass = 0.5
                        if test:
                            print('Test: ', test)
                    else:
                        continue;
                elif (attrValue == 'partial' and attrKey == "partial"):
                    if(originalSequences_set[recordSeq].predictedClass < 0.5):

                        originalSequences_set[recordSeq].predictedClass = 0.5
                        if test:
                            print('Test: ', test)
                    else:
                        continue;
                else:
                    if(not originalSequences_set[recordSeq].predictedClass > 0):
                        originalSequences_set[recordSeq].predictedClass = 0
                    else:
                        continue;
                # elif attrValue == 'partial' and originalSequences_set[recordSeq].predictedClass < 1:
                #     originalSequences_set[recordSeq].predictedClass = 0.5
                # elif attrValue == 'notequal' and originalSequences_set[recordSeq].predictedClass < 0.5:
                #     originalSequences_set[recordSeq].predictedClass = 0

                # originalSequences_set[recordSeq].attrKeys = oriAttrNameArray

                # if len(oriAttrNameArray) > 0:
                #     originalSequences_set[recordSeq].truthClass =  1

                # else:
                #     originalSequences_set[recordSeq].truthClass =  0



                #originalSequences_set[recordSeq].predictedClass = (originalSequences_set[recordSeq].classified + originalSequences_set[recordSeq].attributeKey + originalSequences_set[recordSeq].attributeValue) / 3



                # if testSentence: 
                    # exit()

            

            if(originalSequences_set[recordSeq].predictedClass == 1 and originalSequences_set[recordSeq].truthClass == 0):
                print('you are hre')
                seq = recordSeq
                # print('originalSequences_set[recordSeq]: ', originalSequences_set[recordSeq], ' truthClass: ', originalSequences_set[recordSeq].truthClass)
                testcount = testcount +  1
                # exit()

                # if testcount > 4:
                #     exit()

            # if(testSentence):
            #     exit()
            # if(originalSequences_set[recordSeq].truthClass == 1):
            originalSequences_set[recordSeq].record.append(record)
        
        
        # print("seq: ", originalSequences_set[seq])
        # print("predicted class: ", originalSequences_set[seq].predictedClass)
        # exit()
        
        # for x in originalSequences_set.values():
        #     print('sentence: ', x.sentence)
        #     print('x: ', x.truthClass )

        # exit()

        truthClasses = [x.truthClass for x in originalSequences_set.values()]

        ones = 0
        zeros = 0

        for c in truthClasses:
            if c == 1:
                ones += 1
            else:
                zeros += 1
        print('ones: ', ones, ' zeros: ', zeros)


        pPartial = 0
        pOnes = 0
        pZeros = 0



        for index, c in enumerate(originalSequences_set.values()):
            if c.predictedClass == 1:
                pOnes += 1
            elif c.predictedClass == 0.5:
                pPartial +=1
            else:
                pZeros += 1
                # if c.truthClass == 1:
                #     print('c originalSequences_set: ', c)
                #     if index > 6:
                #         exit()
        print('pOnes: ', pOnes, ' pPartial: ', pPartial, ' pZeros: ', pZeros)


        predictedClasses = [1 if x.predictedClass > 0.4 else 0  for x in originalSequences_set.values()]

        # print(truthClasses)
        # print(predictedClasses)

        onesNow = 0
        zeroNow = 0
        for c in predictedClasses:
            if c == 1:
                onesNow += 1
            else:
                zeroNow += 1

        print('onesNow: ', onesNow, ' zeroNow: ', zeroNow)

        
        print('truthClasses: ', len(truthClasses), ' predictedClasses: ', len(predictedClasses))
        
        print(confusion_matrix(truthClasses, predictedClasses))
        print(classification_report(truthClasses, predictedClasses))

        
        self.writeOutputForClassifiedSequence(originalSequences_set.values(),'Data/TestData/output')


obj = SequenceClassification()
obj.printCacheStats()


#data = obj.readFile('Data/TestData/transcript_vectors2.0_1656680086957.csv') # threshold 0.79
# data = obj.readFile('Data/TestData/transcript_vectors2.0_1657598638418.csv') # threshold 0.0
# data = obj.readFile('Data/TestData/transcript_vectors2.0_1657612391576.csv') # threshold 0.13
# data = obj.readFile('Data/TestData/transcript_vectors2.0_1656680086957.csv') # threshold 0.66
# data = obj.readFile('Data/TestData/labeled/custom/transcript_vectors2.0_1663381884270.csv') # threshold 0.66
# data = obj.readFile('Data/TestData/labeled/custom/transcript_vectors2.0_1663422195130.csv') # threshold 0.79
# data = obj.readFile('Data/TestData/labeled/custom/transcript_vectors2.0_1663450470122.csv') # threshold 0.87
data = obj.readFile('Data/TestData/labeled/custom/transcript_vectors2.0_1663510999286.csv') # threshold 0.6
# data = obj.readFile('Data/TestData/labeled/all-mpnet/transcript_vectors2.0_1663447595489.csv') # threshold 0.5
# data = obj.readFile('Data/TestData/labeled/all-mpnet/transcript_vectors2.0_1663447333081.csv') # threshold 0.45
# data = obj.readFile('Data/TestData/labeled/all-mpnet/transcript_vectors2.0_1663479380236.csv') # threshold 0.49
# data = obj.readFile('Data/TestData/labeled/all-mpnet/transcript_vectors2.0_1663448417000.csv') # threshold temporaray
print('data: ', len(data))

recordWithExtractedValues = obj.getPatternMatchedSentence(data)
print("recordWithExtractedValues: ", len(recordWithExtractedValues))

obj.getTruePossitiveSetnences(recordWithExtractedValues)
