#!./weka_env/bin/python3
import os
import sys
import csv_arff
import part3_attributeReduction
import extract_pixels
import classify
import threading

csv_folder="fer2018/csv"
arff_folder="fer2018/arffs"
reduction_folder="fer2018/reduced_arffs"
transformed_arffs="fer018/transformed_arffs"
pixel_values="pixel_values"

preprocess = False
if(len(sys.argv)==2):
    if sys.argv[1]=="--preprocess":
        preprocess=True

#task1
# filename="fer2018/transformed_arffs/transformed_14.arff"
# testName = "task1_P14"

# #task3
# filename="fer2018/reduced_arffs/fer2018.reduced.arff"
# testName = "task_5_clustering"

# #task5
# filename="fer2018/reduced_arffs/fer2018surprise.reduced.arff"
# testName = "surprise"

#task7
filename="fer2018/transformed_arffs/transformed_14.arff"
testName = "task7_14Pixels_seed_test"


class myThread (threading.Thread):
   def __init__(self, threadID, name, function, args=None):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.function = function
      self.args = args
   def run(self):
      print ("Starting " + self.name)
      self.function(self.args)

def convert_to_arff():

    print("*** Converting CSVs ****\n")
    filenames = os.listdir(csv_folder)

    for csv in filenames:
        filename=os.path.join(csv_folder,csv)
        print("Converting "+filename)
        
        if(csv=="fer2018.csv"):
            csv_converter = csv_arff.Convert(filename,False)
            csv_converter.run()
        else:
            csv_converter = csv_arff.Convert(filename,True)
            csv_converter.run()

def reduce_attr():
    print("\n*** Reducing Arffs ****\n")
    filenames = os.listdir(arff_folder)

    for arff in filenames:
        filename=os.path.join(arff_folder,arff)
        print("Reducing "+filename)
        arff_reducer = part3_attributeReduction.reduce_attributes(filename)
        arff_reducer.run()

def extract():
    print("\n*** Extracting Pixels ****\n")
    filenames = os.listdir(pixel_values)

    for values in filenames:
        filename=os.path.join(pixel_values,values)
        print("Extracting from "+filename)
        extractor = extract_pixels.extract_pix(filename)
        extractor.run()

def run_ibk_crossval(args):
    global filename, testName
    jvm_helper = classify.cw2_helper()

    ibkCls_crossval = classify.cw2_classifier()
    ibkCls_crossval.load_data(filename)
    ibkCls_crossval.run_ibk_crossval("results/"+str(testName))

def run_ibk_split(args):
    global filename, testName
    jvm_helper = classify.cw2_helper()

    ibkCls = classify.cw2_classifier()
    ibkCls.load_data_split(filename,80)
    ibkCls.run_ibk_split("results/"+str(testName))

def run_nb_split(args):
    global filename, testName
    jvm_helper = classify.cw2_helper()

    naiveBayesCls = classify.cw2_classifier()
    naiveBayesCls.load_data_split(filename,80)
    naiveBayesCls.run_naive_bayes_split("results/"+str(testName))

def run_nb_crossval(args):
    global filename, testName
    jvm_helper = classify.cw2_helper()

    naiveBayesCls_crossval = classify.cw2_classifier()
    naiveBayesCls_crossval.load_data(filename)
    naiveBayesCls_crossval.run_naive_bayes_crossval("results/"+str(testName))

def run_bayes_split(parents=1):
    global filename, testName
    jvm_helper = classify.cw2_helper()

    BayesCls_split = classify.cw2_classifier()
    BayesCls_split.load_data_split(filename,80)
    BayesCls_split.run_bayes_split("results/"+str(testName),parents)

    BayesCls_split = classify.cw2_classifier()
    BayesCls_split.load_data_split(filename,80)
    BayesCls_split.run_bayes_hill_split("results/"+str(testName),parents)

    BayesCls_split = classify.cw2_classifier()
    BayesCls_split.load_data_split(filename,80)
    BayesCls_split.run_bayes_tan_split("results/"+str(testName),parents)

def run_simplekm_noclass(args):
    global filename, testName
    jvm_helper = classify.cw2_helper()

    simplek_full = classify.cw2_classifier()
    simplek_full.load_data(filename, True)
    simplek_full.run_cluster_simplek("results/"+str(testName),True, 7)

def run_simplekm_with_class(args):
    global filename, testName
    jvm_helper = classify.cw2_helper()

    simplek_full = classify.cw2_classifier()
    simplek_full.load_data(filename, True)
    simplek_full.run_cluster_simplek("results/"+str(testName),False, 7)

def run_clusters_auto1(num_clusters):
    global filename, testName
    jvm_helper = classify.cw2_helper()

    simplek_full = classify.cw2_classifier()
    simplek_full.load_data(filename)

    clusterList = [2,4,7,10,15]
    for clusterNum in clusterList:
        simplek_full.run_clustering_task7_manual("results/"+str(testName),"weka.clusterers.EM",clusterNum)

def run_clusters_auto2(num_clusters):
    global filename, testName
    jvm_helper = classify.cw2_helper()

    simplek_full = classify.cw2_classifier()
    simplek_full.load_data(filename)

    # simplek_full.run_clustering_task7_auto("results/"+str(testName),"weka.clusterers.Canopy")
    # simplek_full.run_clustering_task7_auto("results/"+str(testName),"weka.clusterers.Cobweb")

def run_clusters_manual(num_clusters):
    global filename, testName
    jvm_helper = classify.cw2_helper()

    simplek_full = classify.cw2_classifier()
    simplek_full.load_data(filename)

    simplek_full.run_clustering_task7_manual("results/"+str(testName),"weka.clusterers.FarthestFirst", num_clusters)
    simplek_full.run_clustering_task7_manual("results/"+str(testName),"weka.clusterers.SimpleKMeans", num_clusters)
    # simplek_full.run_clustering_task7_manual("results/"+str(testName),"weka.clusterers.HierarchicalClusterer", num_clusters)

def test_seeds_manual(args):
    global filename, testName
    jvm_helper = classify.cw2_helper()

    simplek_full = classify.cw2_classifier()
    simplek_full.load_data(filename)

    clusterNum=7
    seedList = [1,10,20,40,50,80,100,150]
    for seed in seedList:
        simplek_full.run_clustering_task7_manual("results/"+str(testName),"weka.clusterers.EM", clusterNum, seed)
        simplek_full.run_clustering_task7_manual("results/"+str(testName),"weka.clusterers.FarthestFirst", clusterNum, seed)


def run_classifiers():

    jvm_helper = classify.cw2_helper(False)
    jvm_helper.cleanup()

    threads = []

    #TASK 1
    # Create new threads
    # thread1 = myThread(1, "IBK-Cross-Val", run_ibk_crossval)
    # thread2 = myThread(2, "IBK-Split", run_ibk_split)
    # thread3 = myThread(3, "run_nb_split", run_nb_split)
    # thread4 = myThread(4, "run_nb_crossval", run_nb_crossval)

    #TASK3
    # thread1 = myThread(1, "run_bayes_split", run_bayes_split, (1))
    # thread2 = myThread(2, "run_bayes_split", run_bayes_split, (3))
    # thread3 = myThread(3, "run_bayes_split", run_bayes_split, (5))

    #TASK5
    # thread1 = myThread(1, "run_simplekm_noclass", run_simplekm_noclass)
    # thread2 = myThread(2, "run_simplekm_with_class", run_simplekm_with_class)

    #TASK7
    # thread1 = myThread(1, "run_clusters_auto1", run_clusters_auto1)
    # thread1 = myThread(2, "run_clusters_auto2", run_clusters_auto2)

    # thread1 = myThread(1, "run_clusters_manual", run_clusters_manual, 4)
    # thread2 = myThread(2, "run_clusters_manual", run_clusters_manual, (15))
    # thread3 = myThread(3, "run_clusters_manual", run_clusters_manual, (7))
    # thread4 = myThread(4, "run_clusters_manual", run_clusters_manual, (10))

    thread1 = myThread(4, "run_seeds_manual", test_seeds_manual, (0))

    # Start new Threads
    thread1.start()
    # thread2.start()
    # thread3.start()
    # thread4.start()

    # Add threads to thread list
    threads.append(thread1)
    # threads.append(thread2)
    # threads.append(thread3)
    # threads.append(thread4)

    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    jvm_helper.cleanup()
    print ("Exiting Main Thread")

try:
    if(preprocess):
        print("***** Preprocessing Data ******")
        convert_to_arff()
        reduce_attr()
        extract()
    else:
        run_classifiers()

except Exception as e:
    print(e)


    