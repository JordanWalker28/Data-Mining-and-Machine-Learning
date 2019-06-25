#!/bin/python3

import sys
import os
import csv

output_folder="fer2018/arffs"

class Convert:

    def __init__(self,filename, skip):
        self.command = skip
        self.filename = filename
        self.emotion_name="Happy"

    def output_header(self, output_file):
        emotions_full = "{angry,disgust,fear,happy,neutral,sad,suprise}"

        output_file.write("@Relation faces\n")
        
        if(self.command):
            emotions_skip = "{NotHappy,"+self.emotion_name+"}"
            emotion_string = str.format("@ATTRIBUTE emotion {0}\n",emotions_skip)
        else:
            emotion_string = str.format("@ATTRIBUTE emotion {0}\n",emotions_full)

        output_file.write(emotion_string)

        for x in range(2304):
            output_file.write("@ATTRIBUTE pixel"+str(x)+" numeric\n")
        output_file.write("@DATA\n")

    def open_files(self,filename):
        input_file = open(filename,"r")
        output_file = os.path.join(output_folder,os.path.basename(filename[:-4]+".arff"))
        try:
            output_file = open(output_file,"x")
        except:
            os.remove(output_file)
            print("Removed exisiting file\n")
            output_file = open(output_file,"a")
            
        return input_file, output_file

    def convert_emotion(self,emotion):
        dict = {'0' : "angry", '1' : "disgust", '2' : "fear", '3' : "happy", '4' : "sad", '5' : "suprise", '6' : "neutral"}
        return dict[emotion]

    def get_convert_data(self,input_file):
        data = []

        csv_data = csv.reader(input_file, delimiter=',')
        skip_firstline = True

        for row in csv_data:
            if skip_firstline:
                skip_firstline = False
                continue

            if(self.command):
                if(row[-1]=="NotHappy"):
                    return_row = "NotHappy"
                else:
                    return_row = self.emotion_name
            else:
                return_row = self.convert_emotion(row[0])

            if(self.command):
                pixel_array = row[:-1]
            else:
                pixel_array = row[1].split(" ")

            for pixel in pixel_array:
                return_row += ","+str(int(pixel)/255)
            
            return_row+="\n"
            data.append(return_row)
        return data

    def write_data(self, output_file, data):
        try:
            self.output_header(output_file)
            for row in data:
                output_file.write(row)
            print("*** Conversion Compelete ***\n")
        except Exception as e:
            print("Exception while writing data.\n")
            print(e)

    def run(self):
        try:
            csv_file, arff_file = self.open_files(self.filename)
            converted_data = self.get_convert_data(csv_file)
            self.write_data(arff_file, converted_data)
        finally:
            csv_file.close
            arff_file.close