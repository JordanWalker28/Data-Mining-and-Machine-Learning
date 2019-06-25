#!/usr/bin/python3
''' F20DL
Coursework 1
Part 3: Reducing Number of Attributes
Author: Cameron McBride
Group: 9
'''

import sys

if (len(sys.argv) != 2): # check correct number of args given
	print("Provide one input .arff file as a command-line argument")
	sys.exit()

def newFile(oldName): # returns new filename for reduced attributes
	newName = oldName[0:(len(oldName)-5)]+".reduced.arff" # new file name
	return newName

def calculate(pixelList): # Returns average value of a list of pixels
	total = 0
	for i in pixelList: 
		total += i # add all pixel values together
	
	total = total/len(pixelList)
	return str(int(total)) # divide by number of pixels for average


def iterate(filenameIn, filenameOut): # iterate through pixels of input file
	input = open(filenameIn, 'r')
	output = open(filenameOut, 'w')
	count = 0

	line1 = input.readline()
	line2 = input.readline()

	output.write(line1)
	output.write(line2)
	
	for i in range(576):
		output.write("@ATTRIBUTE pixel" + str(i) +" numeric\n")
	
	for i in input: # iterate through input file to find line before data begins
		if i == "@DATA\n":
			break
	
	output.write("@DATA\n")

	for input_line in input:
		#line = input.readline() # read next line (corresponds to one image)
		line = input_line
		datum = line.split(",") # Split data

		output.write(datum[0]+",") # datum[0] is the emotion String

		a = 1 # first pixel of square
		b = a + 1 # second pixel of square
		c = 49 # third pixel
		d = c + 1 # fourth

		for i in range(24):
			for j in range(24):
				pixelList = [int(datum[a]), int(datum[b]), int(datum[c]), int(datum[d])]
				squareValue = calculate(pixelList)
				# TODO write squareValue to output file followed by comma unless end of line
				output.write(squareValue)
				if (d < 2304): 
					output.write(",")
				else: 
					output.write("\n")
					# print("Instance number " + str(count))
					count+=1

				a+=2 # Increment indices by 2 for next square group of pixels
				b+=2 
				c+=2 
				d+=2

			a+=48 # Increment indices by 48 for next two rows of pixels
			b+=48 
			c+=48 
			d+=48

	input.close()
	output.close()

	print("\n***Reduction Complete***\n")

'''-------------------------------------------'''


inputFile = sys.argv[1]
outputFile = newFile(inputFile)
iterate(inputFile, outputFile)