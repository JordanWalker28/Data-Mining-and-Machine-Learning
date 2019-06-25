import os
import random
import fileinput
import linecache

def createHeader():
    f = open("fer2018.arff","r")
    copy = open("header.arff", "w")
    for line in f:
        if "DATA" in line:
            copy.write(line)
            break
        copy.write(line)
    f.close()

def createData():
    filenames = ['fer2018angry.arff', 'fer2018disgust.arff', 'fer2018surprise.arff'
                 , 'fer2018fear.arff', 'fer2018happy.arff', 'fer2018neutral.arff', 'fer2018sad.arff']
    with open("data.arff", 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    if "DATA" in line:
                        for line in infile:
                            outfile.write(line)


def selectData():
    num_lines = sum(1 for line in open('data.arff'))
    print(str(num_lines))
    size = int(input("Select Data Size: "))
    f = open("data.arff", "r")
    lines = f.readlines()
    newDF = open("newData.arff", "w")
    always_print = False
    j = 0
    while(j < size):
        print(lines[j])
        j = j + 1
    i = 0
    while( i < 3):
        newDF.write("% \n")
        i = i + 1
    f.close()

def createNewFile(filename):

    filenames = ['header.arff', 'newData.arff']
    with open(filename + ".arff", 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                outfile.write("\n")

def main():
    filename = (input("Please enter file name: "))
    createHeader()
    print("***attributes generated***")
    createData()
    print("***data gathered***")
    selectData()
    print("***data selected***")
    createNewFile(filename)
    print("***" + filename +".arff has been generated***")


main()
    
