#!/bin/python3

import sys
import os
import csv
from random import shuffle

if len(sys.argv) >= 2:
    input_filename = sys.argv[1]
else:
    print("You must enter the input filename.")
    exit(1)

def output_header(output_file,pixel_range):
    emotions_full = "{angry,disgust,fear,happy,neutral,sad,surprise}"

    output_file.write("@Relation faces\n")

    emotion_string = str.format("@ATTRIBUTE emotion {0}\n",emotions_full)

    output_file.write(emotion_string)

    for x in range(pixel_range):
        output_file.write("@ATTRIBUTE pixel"+str(x)+" numeric\n")
    output_file.write("@DATA\n")

def open_files(filename):
    number=0
    try:
        input_file = open(filename,"r")
        pixels_to_extract = input_file.readline().split(",")
        number = len(pixels_to_extract)
    except:
        print("Could not open input file.")

    output_file = "fer2018/transformed_arffs/transformed_"+str(number)+".arff"

    try:
        output_file = open(output_file,"x")
    except:
        os.remove(output_file)
        print("Removed exisiting file\n")
        output_file = open(output_file,"a")
        
    return input_file, output_file, pixels_to_extract

def extract_pixels(input_file, pixels_to_extract):
    data = []

    for input_line in input_file:
        if input_line == "@DATA\n":
            break

    for input_line in input_file:

        input_line_split = input_line.split(",")

        return_row = input_line_split[0]

        if(return_row=="other"):
            continue

        for pixel in pixels_to_extract:
            return_row += ","+str(input_line_split[int(pixel)+1])
        
        return_row+="\n"
        data.append(return_row)
    return data

def write_data(output_file, data, pixel_range):
    
    shuffle(data)

    try:
        output_header(arff_file,pixel_range)

        for row in data:
            output_file.write(row)
        print("*** Pixel Extraction Compelete ***\n")
    except Exception as e:
        print("Exception while writing data.\n")
        print(e)

def extract_many(input_file,output_file,pixels_to_extract):

    data=[]

    for filename in input_file:
        try:
            input_file = open(filename.rstrip(),"r")
            data += extract_pixels(input_file,pixels_to_extract)
        except Exception as e:
            print("Error during extraction process.")
            print("Check directory seperator symbol in input file list (Mac/Linux/Windows)")
            print(e)

    return data


#Main

try:
    input_file, arff_file, pixels_to_extract = open_files(input_filename)
    data = extract_many(input_file,arff_file,pixels_to_extract)
    write_data(arff_file, data, len(pixels_to_extract))
finally:
    input_file.close
    arff_file.close