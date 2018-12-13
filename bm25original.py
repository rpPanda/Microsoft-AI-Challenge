import math
import pickle
import string
import string
from nltk.corpus import stopwords       
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import time
import Stemmer
#Initialize Global variables 
docIDFDict = {}
avgDocLength = 0
stop_words = set(stopwords.words("english"))
# print(cachedStopWords)
intab = string.punctuation
lenstr = len(string.punctuation)
outtab = " " * lenstr
trantab = str.maketrans(intab, outtab)
ps = PorterStemmer()
start=time.time()
stemmer=Stemmer.Stemmer('english')


def GetCorpus(inputfile,corpusfile):
    f = open(inputfile,"r",encoding="utf-8")
    fw = open(corpusfile,"w",encoding="utf-8")
    i=0
    for line in f:
        i=i+1
        if i%10000 == 0:
            print(i)
            # print(i)
            print(time.time()-start)
        if len(line.strip().lower().split('\t')) > 1 :
            # print(line.strip().lower().split('\t'))
            passage = line.strip().lower().split("\t")[2] # +line.strip().lower().split("\t")[3]
            passage = passage.translate(trantab)
            passage=' '.join(passage.split())
            d=passage.split(' ')
            stemmed=stemmer.stemWords(d)
            for w in stemmed:
                passage=passage+w+" "

            # word_tokens = word_tokenize(passage) 
            # passage=""
            # for word in word_tokens:
                # if(word not in stop_words):
                    # passage=passage+" "+word
            # print(passage)
            # passage = ''.join([word for word in passage.split() if word not in cachedStopWords])

        # passage='hi'
        fw.write(passage+"\n")
    f.close()
    fw.close()



# The following IDF_Generator method reads all the passages(docs) and creates Inverse Document Frequency(IDF) scores for each unique word using below formula 
# IDF(q_i) = log((N-n(q_i)+0.5)/(n(q_i)+0.5)) where N is the total number of documents in the collection and n(q_i) is the number of documents containing q_i
# After finding IDF scores for all the words, The IDF dictionary will be saved in "docIDFDict.pickle" file in the current directory

def IDF_Generator(corpusfile, delimiter=' ', base=math.e) :

    global docIDFDict,avgDocLength

    docFrequencyDict = {}       
    numOfDocuments = 0   
    totalDocLength = 0

    for line in open(corpusfile,"r",encoding="utf-8") :
        doc = line.strip().split(delimiter)
        # # doc[7]=doc[7].translate(string.punctuation)
        # print(doc)
        totalDocLength += len(doc)

        doc = list(set(doc)) # Take all unique words

        for word in doc : #Updates n(q_i) values for all the words(q_i)
            if word not in docFrequencyDict :
                docFrequencyDict[word] = 0
            docFrequencyDict[word] += 1

        numOfDocuments = numOfDocuments + 1
        if (numOfDocuments%50000==0):
            print(numOfDocuments)                

    for word in docFrequencyDict:  #Calculate IDF scores for each word(q_i)
        docIDFDict[word] = math.log((numOfDocuments) / (docFrequencyDict[word]), base) #Why are you considering "numOfDocuments - docFrequencyDict[word]" instead of just "numOfDocuments"

    avgDocLength = totalDocLength / numOfDocuments

     
    pickle_out = open("docIDFDict.pickle","wb") # Saves IDF scores in pickle file, which is optional
    pickle.dump(docIDFDict, pickle_out)
    pickle_out.close()


    print("NumOfDocuments : ", numOfDocuments)
    print("AvgDocLength : ", avgDocLength)



#The following GetBM25Score method will take Query and passage as input and outputs their similarity score based on the term frequency(TF) and IDF values.
def GetBM25Score(Query, Passage, k1=1.2, b=0.75, delimiter=' ') :
    
    global docIDFDict,avgDocLength

    query_words= Query.strip().lower().translate(trantab)
    word_tokens = word_tokenize(query_words) 
    query_words = [w for w in word_tokens if not w in stop_words] 
    passage_words=Passage.strip().lower().translate(trantab)
    word_tokens = word_tokenize(passage_words) 
    passage_words = [w for w in word_tokens if not w in stop_words] 
    query_words=stemmer.stemWords(query_words)
    # for w in query_words:
    #     query_words_stem.append(ps.stem(w))
    # query_words=query_words_stem
    # print(query_words)
    passage_words=Passage.strip().lower().translate(trantab)
    word_tokens = word_tokenize(passage_words) 
    passage_words = [w for w in word_tokens if not w in stop_words] 
    passage_words=stemmer.stemWords(passage_words)
    # passage_words_stem=[]
    # for w in passage_words:
    #     passage_words_stem.append(ps.stem(w))
    # passage_words=passage_words_stem
    
    passage_words=Passage.strip().lower().translate(trantab)
    word_tokens = word_tokenize(passage_words) 
    passage_words = [w for w in word_tokens if not w in stop_words] 
    # passage_words = passage_words.split(delimiter)
    # print(passage_words)
    passageLen = len(passage_words)
    docTF = {}
    for word in set(query_words):   #Find Term Frequency of all query unique words
        docTF[word] = passage_words.count(word)
    commonWords = set(query_words) & set(passage_words)
    tmp_score = []
    for word in commonWords :   
        numer = (docTF[word] * (k1+1))   #Numerator part of BM25 Formula
        denom = ((docTF[word]) + k1*(1 - b + b*passageLen/avgDocLength)) #Denominator part of BM25 Formula 
        if(word in docIDFDict) :
            tmp_score.append(docIDFDict[word] * (numer / denom  +1))

    score = sum(tmp_score)
    return score

#The following line reads each line from testfile and extracts query, passage and calculates BM25 similarity scores and writes the output in outputfile
def RunBM25OnEvaluationSet(testfile,outputfile):

    lno=0
    tempscores=[]  #This will store scores of 10 query,passage pairs as they belong to same query
    f = open(testfile,"r",encoding="utf-8")
    fw = open(outputfile,"w",encoding="utf-8")
    for line in f:
        tokens = line.strip().lower().split("\t")
        Query = tokens[1]
        Passage = tokens[2]
        score = GetBM25Score(Query,Passage) 
        tempscores.append(score)
        lno+=1
        if(lno%10==0):
            tempscores = [str(s) for s in tempscores]
            scoreString = "\t".join(tempscores)
            qid = tokens[0]
            fw.write(qid+"\t"+scoreString+"\n")
            tempscores=[]
        if(lno%5000==0):
            print(lno)
    print(lno)
    f.close()
    fw.close()


if __name__ == '__main__' :

    inputFileName = "Data.tsv"   # This file should be in the following format : queryid \t query \t passage \t label \t passageid
    testFileName = "eval1_unlabelled.tsv"  # This file should be in the following format : queryid \t query \t passage \t passageid # order of the query
    corpusFileName = "corpus.tsv" 
    outputFileName = "answer.tsv"

    GetCorpus(inputFileName,corpusFileName)    # Gets all the passages(docs) and stores in corpusFile. you can comment this line if corpus file is already generated
    print("Corpus File is created.")
    IDF_Generator(corpusFileName)   # Calculates IDF scores. 
    #RunBM25OnTestData(testFileName,outputFileName)
    print("IDF Dictionary Generated.")
    RunBM25OnEvaluationSet(testFileName,outputFileName)
    print("Submission file created. ")
    # print(string.punctuation)
