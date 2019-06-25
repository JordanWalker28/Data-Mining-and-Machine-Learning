import sys
import os.path
import random
import fileinput
import linecache

def createNewFile():
    filenames = ['header.arff', 'shuffled.arff']
    with open("shuffled" + sys.argv[1], 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                outfile.write("\n")

def randomData():
    f = open("data.arff", "r")
    lines = f.readlines()
    outputFile = open("shuffled.arff", 'w')
    
    
    i = 0
    seenList = []
    num_lines = sum(1 for line in open("data.arff",'r'))

    while(i < num_lines):
        randomN = random.randint(1,num_lines)
        seenList.append(randomN)
        if randomN in seenList == True:
            while(randomN in seenList != True):
                randomN += 1
        else:
            i += 1

    for x in seenList:
        outputFile.write(lines[seenList[x]])

    text = "% \n"
    outputFile.write(text * 3)
    f.close()


def createHeader():
    f = open(sys.argv[1],'r')
    copy = open("header.arff", "w")
    for line in f:
        if "DATA" in line:
            copy.write(line)
            break
        copy.write(line)
    f.close()

def createData():
    
    f = open(sys.argv[1],'r')
    copy = open("data.arff", "w")
    for line in f:
        if "DATA" in line:
            for line in f:
                copy.write(line)
    f.close()


def main():
    if (len(sys.argv) != 1):
        print("correct")
    else:
        print("Error...  you did not enter a filename")
        str = input("please enter an .arff file you wish to randomise: ")
        if(os.path.isfile(str) == True):
            sys.argv.append( "random"+str)
        else:
            while(os.path.isfile(str) != True):
                print ("file does not exist try again")
                str = input("please enter an .arff file you wish to randomise: ")

    createHeader()
    print("*** HEADER CREATED ***")
    createData()
    print("*** DATA CREATED ***")
    randomData()
    print("*** DATA RANDOMISED ***")
    createNewFile()
    print("*** NEW FILE CREATED ***")

    os.remove('data.arff')
    os.remove('header.arff')
    os.remove('shuffled.arff')



main()
