#!/bin/python3

import sys
import os
import csv

CMD_1 = "--skip"

if len(sys.argv) >= 3:
    command = sys.argv[1]
    filename = sys.argv[2]
    filename2 = os.path.basename(filename)
    emotion_name=filename2[7:-4]

    if(command!=CMD_1):
        print("Available commands: "+CMD_1)

elif len(sys.argv) >= 2:
    command=""
    filename = sys.argv[1]
else:
    print("You must enter a file name to covnert.")
    exit(1)

def output_header(output_file):
    emotions_full = "{angry,disgust,fear,happy,neutral,sad,suprise}"

    output_file.write("@Relation faces\n")
    
    if(command==CMD_1):
        emotions_skip = "{other,"+emotion_name+"}"
        emotion_string = str.format("@ATTRIBUTE emotion {0}\n",emotions_skip)
    else:
        emotion_string = str.format("@ATTRIBUTE emotion {0}\n",emotions_full)

    output_file.write(emotion_string)

    for x in range(2304):
        output_file.write("@ATTRIBUTE pixel"+str(x)+" numeric\n")
    output_file.write("@DATA\n")

def open_files(filename):
    input_file = open(filename,"r") 
    output_file = filename[:-4]+".arff"
    try:
        output_file = open(output_file,"x")
    except:
        os.remove(output_file)
        print("Removed exisiting file\n")
        output_file = open(output_file,"a")
        
    return input_file, output_file

def convert_emotion(emotion):
	dict = {'0' : "angry", '1' : "disgust", '2' : "fear", '3' : "happy", '4' : "sad", '5' : "suprise", '6' : "neutral"}
	return dict[emotion]

def get_convert_data(input_file):
    data = []

    csv_data = csv.reader(input_file, delimiter=',')
    skip_firstline = True

    for row in csv_data:
        if skip_firstline:
            skip_firstline = False
            continue

        if(command==CMD_1):
            if(row[0]=="0"):
                return_row = "other"
            else:
                return_row = emotion_name
        else:
            return_row = convert_emotion(row[0])

        pixel_array = row[1].split(" ")

        for pixel in pixel_array:
            return_row += ","+pixel
        
        return_row+="\n"
        data.append(return_row)
    return data

def write_data(output_file, data):
    
    try:
        output_header(arff_file)
        for row in data:
            output_file.write(row)
        print("*** Conversion Compelete ***\n")
    except Exception as e:
        print("Exception while writing data.\n")
        print(e)
#Main

try:
    csv_file, arff_file = open_files(filename)
    converted_data = get_convert_data(csv_file)
    write_data(arff_file, converted_data)
finally:
    csv_file.close
    arff_file.close