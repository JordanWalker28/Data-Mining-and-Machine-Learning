#!/usr/bin/python3
''' F20DL
Coursework 1
Part 3: Reducing Number of Attributes
Author: Cameron McBride
Group: 9
'''

import sys
import os

class reduce_attributes():

	def __init__(self,filename):
		self.filename = filename

	def newFile(self,oldName): # returns new filename for reduced attributes
		inner_directory = os.path.dirname(oldName)
		outer_directory = os.path.dirname(inner_directory)
		oldName = os.path.basename(oldName)
		newName = oldName[0:(len(oldName)-5)]+".reduced.arff" # new file name
		newName = os.path.join(os.path.join(outer_directory,"reduced_arffs"),newName)
		# print("File output to: "+newName)
		return newName

	def calculate(self, pixelList): # Returns average value of a list of pixels
		total = 0
		for i in pixelList: 
			total += i # add all pixel values together
		
		total = total/len(pixelList)
		return str(total) # divide by number of pixels for average


	def iterate(self, filenameIn, filenameOut): # iterate through pixels of input file
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
					pixelList = [float(datum[a]), float(datum[b]), float(datum[c]), float(datum[d])]
					squareValue = self.calculate(pixelList)
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

	def run(self):
		outputFile = self.newFile(self.filename)
		self.iterate(self.filename, outputFile)


