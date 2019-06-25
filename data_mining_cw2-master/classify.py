import os
import sys
import traceback
import weka.core.jvm as jvm
from weka.core.classes import Random
from weka.core.converters import Loader, Instances
from weka.classifiers import Classifier, Evaluation
import time
from weka.filters import Filter
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from weka.clusterers import Clusterer, ClusterEvaluation

class cw2_classifier():
    def __init__(self):
        pass
        # jvm.start()

    def load_data(self, filename, filter=False):
        self.filename = filename
        print("\nLoading dataset: " + filename)
        loader = Loader(classname="weka.core.converters.ArffLoader")
        data = loader.load_file(filename)
        data.class_is_first()
        if(filter):
            data = self.filter_data(data)
        self.training_data = data

    def load_data_split(self, filename, validation_split, filter=False):
        self.validation_split = validation_split    
        self.filename = filename
        print("\nLoading dataset: " + filename)
        loader = Loader(classname="weka.core.converters.ArffLoader")
        data = loader.load_file(filename)
        data.class_is_first()
        if(filter):
            data = self.filter_data(data)
        train, test = data.train_test_split(self.validation_split, Random(1))
        self.training_data = train
        self.testing_data = test

    def run_naive_bayes_split(self, output_directory):
        # build classifier
        print("\nBuilding Classifier on training data.")
        buildTimeStart=time.time()
        cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
        cls.build_classifier(self.training_data)

        resultsString = ""
        resultsString = self.print_both(str(cls),resultsString)

        buildTimeString = "NB Split Classifier Built in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Evaluate Classifier
        resultsString = self.print_both("\nEvaluating on test data.",resultsString)

        buildTimeStart=time.time()
        evl=Evaluation(self.training_data)
        evl.test_model(cls, self.testing_data)

        resultsString = self.print_both(str(evl.summary()),resultsString)
        resultsString = self.print_both(str(evl.class_details()),resultsString)
        resultsString = self.print_both(str(evl.confusion_matrix),resultsString)
        buildTimeString = "\nNB Split Classifier Evaluated in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Save Results and Cleanup
        self.save_results("Naive_Bayes",resultsString,output_directory)
    
    def run_naive_bayes_crossval(self, output_directory):
        # build classifier
        print("\nBuilding Classifier on training data.")
        buildTimeStart=time.time()
        cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
        cls.build_classifier(self.training_data)

        resultsString = ""
        resultsString = self.print_both(str(cls),resultsString)

        buildTimeString = "NB Cross Eval Classifier Built in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Evaluate Classifier
        resultsString = self.print_both("\nCross Evaluating on test data.",resultsString)

        buildTimeStart=time.time()
        evl = Evaluation(self.training_data)
        evl.crossvalidate_model(cls, self.training_data, 10, Random(1))

        resultsString = self.print_both(str(evl.summary()),resultsString)
        resultsString = self.print_both(str(evl.class_details()),resultsString)
        resultsString = self.print_both(str(evl.confusion_matrix),resultsString)
        buildTimeString = "\nNB Cross Eval Classifier Evaluated in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Save Results and Cleanup
        self.save_results("Naive_Bayes_Crossval",resultsString,output_directory)

    def run_bayes_split(self, output_directory, parents=1):
        # build classifier
        print("\nBuilding Bayes Classifier on training data. Parents = "+str(parents)+"\n")
        buildTimeStart=time.time()
        cls = Classifier(classname="weka.classifiers.bayes.BayesNet", options=["-D","-Q", "weka.classifiers.bayes.net.search.local.K2", "--", "-P", ""+str(parents),"-S", "BAYES", "-E", "weka.classifiers.bayes.net.estimate.SimpleEstimator", "--", "-A", "0.5"])
        cls.build_classifier(self.training_data)

        resultsString = ""
        resultsString = self.print_both(str(cls),resultsString)

        buildTimeString = "Bayes Split Classifier Built in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Evaluate Classifier
        resultsString = self.print_both("\nEvaluating on test data.",resultsString)

        buildTimeStart=time.time()
        evl=Evaluation(self.training_data)
        evl.test_model(cls, self.testing_data)

        resultsString = self.print_both(str(evl.summary()),resultsString)
        resultsString = self.print_both(str(evl.class_details()),resultsString)
        resultsString = self.print_both(str(evl.confusion_matrix),resultsString)
        buildTimeString = "\nBayes Split Classifier Evaluated in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Save Results and Cleanup
        self.save_results("Bayes_K_P"+str(parents)+"_",resultsString,output_directory)
        self.save_results("Bayes_K_P"+str(parents)+"_Graph",cls.graph,output_directory, True)

    def run_bayes_hill_split(self, output_directory, parents=1):
        # build classifier
        print("\nBuilding Bayes Classifier on training data. Parents = "+str(parents)+"\n")
        buildTimeStart=time.time()
        cls = Classifier(classname="weka.classifiers.bayes.BayesNet", options=["-D","-Q", "weka.classifiers.bayes.net.search.local.HillClimber", "--", "-P", ""+str(parents),"-S", "BAYES", "-E", "weka.classifiers.bayes.net.estimate.SimpleEstimator", "--", "-A", "0.5"])
        cls.build_classifier(self.training_data)

        resultsString = ""
        resultsString = self.print_both(str(cls),resultsString)

        buildTimeString = "Bayes Split Classifier Built in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Evaluate Classifier
        resultsString = self.print_both("\nEvaluating on test data.",resultsString)

        buildTimeStart=time.time()
        evl=Evaluation(self.training_data)
        evl.test_model(cls, self.testing_data)

        resultsString = self.print_both(str(evl.summary()),resultsString)
        resultsString = self.print_both(str(evl.class_details()),resultsString)
        resultsString = self.print_both(str(evl.confusion_matrix),resultsString)
        buildTimeString = "\nBayes Split Classifier Evaluated in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Save Results and Cleanup
        self.save_results("Bayes_Hill_P"+str(parents)+"_",resultsString,output_directory)
        self.save_results("Bayes_Hill_P"+str(parents)+"_Graph",cls.graph,output_directory, True)

    def run_bayes_tan_split(self, output_directory, parents=1):
        # build classifier
        print("\nBuilding Bayes Classifier on training data. Parents = "+str(parents)+"\n")
        buildTimeStart=time.time()
        cls = Classifier(classname="weka.classifiers.bayes.BayesNet", options=["-D","-Q", "weka.classifiers.bayes.net.search.local.TAN", "--","-S", "BAYES", "-E", "weka.classifiers.bayes.net.estimate.SimpleEstimator", "--", "-A", "0.5"])
        cls.build_classifier(self.training_data)

        resultsString = ""
        resultsString = self.print_both(str(cls),resultsString)

        buildTimeString = "Bayes Split Classifier Built in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Evaluate Classifier
        resultsString = self.print_both("\nEvaluating on test data.",resultsString)

        buildTimeStart=time.time()
        evl=Evaluation(self.training_data)
        evl.test_model(cls, self.testing_data)

        resultsString = self.print_both(str(evl.summary()),resultsString)
        resultsString = self.print_both(str(evl.class_details()),resultsString)
        resultsString = self.print_both(str(evl.confusion_matrix),resultsString)
        buildTimeString = "\nBayes Split Classifier Evaluated in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Save Results and Cleanup
        self.save_results("Bayes_TAN_P"+str(parents)+"_",resultsString,output_directory)
        self.save_results("Bayes_TAN_P"+str(parents)+"_Graph",cls.graph,output_directory, True)

    def run_ibk_split(self, output_directory):
        # build classifier
        print("\nBuilding Classifier on training data.")
        buildTimeStart=time.time()
        cls = Classifier(classname="weka.classifiers.lazy.IBk", options=["-K", "3", "-W", "0", "-A", "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""])
        cls.build_classifier(self.training_data)

        resultsString = ""
        resultsString = self.print_both(str(cls),resultsString)

        buildTimeString = "IBK Split Classifier Built in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Evaluate Classifier
        resultsString = self.print_both("\nEvaluating on test data.",resultsString)

        buildTimeStart=time.time()
        evl=Evaluation(self.training_data)
        evl.test_model(cls, self.testing_data)

        resultsString = self.print_both(str(evl.summary()),resultsString)
        resultsString = self.print_both(str(evl.class_details()),resultsString)
        resultsString = self.print_both(str(evl.confusion_matrix),resultsString)
        buildTimeString = "\nIBK Split Classifier Evaluated in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Save Results and Cleanup
        self.save_results("IBK",resultsString,output_directory)

    def run_ibk_crossval(self, output_directory):
        # build classifier
        print("\nBuilding Classifier on training data.")
        buildTimeStart=time.time()
        cls = Classifier(classname="weka.classifiers.lazy.IBk", options=["-K", "3", "-W", "0", "-A", "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""])
        cls.build_classifier(self.training_data)

        resultsString = ""
        resultsString = self.print_both(str(cls),resultsString)

        buildTimeString = "IBK Cross Eval Classifier Built in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Evaluate Classifier
        resultsString = self.print_both("\nCross Evaluating on test data.",resultsString)

        buildTimeStart=time.time()
        evl = Evaluation(self.training_data)
        evl.crossvalidate_model(cls, self.training_data, 10, Random(1))

        resultsString = self.print_both(str(evl.summary()),resultsString)
        resultsString = self.print_both(str(evl.class_details()),resultsString)
        resultsString = self.print_both(str(evl.confusion_matrix),resultsString)
        buildTimeString = "\nIBK Cross Eval Classifier Evaluated in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)
        
        #Save Results and Cleanup
        self.save_results("IBK_Crossval",resultsString,output_directory)

    def run_cluster_simplek(self, output_directory, exc_class=False, num_clusters=7):
        data = Instances.copy_instances(self.training_data)
        data.no_class()
        data.delete_first_attribute()

        # build a clusterer and output model
        print("\nBuilding Clusterer on training data.")
        buildTimeStart=time.time()
        clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", ""+str(num_clusters)])
        clusterer.build_clusterer(data)

        resultsString = ""
        resultsString = self.print_both(str(clusterer),resultsString)

        buildTimeString = "Clusterer Built in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)

        #Evaluate Clusterer
        resultsString = self.print_both("\nClustering data.",resultsString)

        buildTimeStart=time.time()

        clsexc = ""
        if(exc_class):
            # no class attribute
            clsexc = "_NO_Class"
            evl = ClusterEvaluation()
            evl.set_model(clusterer)
            evl.test_model(data)
        else:
            # classes to clusters
            evl = ClusterEvaluation()
            evl.set_model(clusterer)
            evl.test_model(self.training_data)

        resultsString = self.print_both("\nCluster results\n",resultsString)
        resultsString = self.print_both(str(evl.cluster_results),resultsString)

        resultsString = self.print_both("\nClasses to clusters\n",resultsString)
        resultsString = self.print_both(str(evl.classes_to_clusters),resultsString)

        buildTimeString = "\nClustered data in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)

        #Save Results and Cleanup
        self.save_results("SimpleKM"+clsexc+"_",resultsString,output_directory)

    def run_clustering_task7_auto(self, output_directory, clusterer_name):
        data = Instances.copy_instances(self.training_data)
        data.no_class()
        data.delete_first_attribute()

        clusterer_name_short = clusterer_name.replace("weka.clusterers.","")

        # build a clusterer and output model
        print("\nBuilding "+clusterer_name_short+" Clusterer on training data.")
        buildTimeStart=time.time()
        # clusterer = Clusterer(classname=clusterer_name, options=["-num-slots", "4"])
        clusterer = Clusterer(classname=clusterer_name)
        clusterer.build_clusterer(data)

        resultsString = ""
        resultsString = self.print_both(str(clusterer),resultsString)

        buildTimeString = "Clusterer Built in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)

        #Evaluate Clusterer
        resultsString = self.print_both("\nClustering data.",resultsString)

        buildTimeStart=time.time()

        evl = ClusterEvaluation()
        evl.set_model(clusterer)
        evl.test_model(self.training_data)

        resultsString = self.print_both("\nCluster results\n",resultsString)
        resultsString = self.print_both(str(evl.cluster_results),resultsString)

        resultsString = self.print_both("\nClasses to clusters\n",resultsString)
        resultsString = self.print_both(str(evl.classes_to_clusters),resultsString)

        buildTimeString = "\nClustered data in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)

        #Save Results and Cleanup
        self.save_results(clusterer_name_short+"_",resultsString,output_directory)

    def run_clustering_task7_manual(self, output_directory, clusterer_name, num_clusters, seed=10):
        data = Instances.copy_instances(self.training_data)
        data.no_class()
        data.delete_first_attribute()
        
        clusterer_name_short = clusterer_name.replace("weka.clusterers.","")
        # build a clusterer and output model
        print("\nBuilding "+clusterer_name_short+" Clusterer on training data.")
        buildTimeStart=time.time()
        clusterer = Clusterer(classname=clusterer_name, options=["-N", ""+str(num_clusters), "-S", ""+str(seed)])
        clusterer.build_clusterer(data)

        resultsString = ""
        resultsString = self.print_both(str(clusterer),resultsString)

        buildTimeString = "Clusterer Built in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)

        #Evaluate Clusterer
        resultsString = self.print_both("\nClustering data.",resultsString)

        buildTimeStart=time.time()

        evl = ClusterEvaluation()
        evl.set_model(clusterer)
        evl.test_model(self.training_data)

        resultsString = self.print_both("\nCluster results\n",resultsString)
        resultsString = self.print_both(str(evl.cluster_results),resultsString)

        resultsString = self.print_both("\nClasses to clusters\n",resultsString)
        resultsString = self.print_both(str(evl.classes_to_clusters),resultsString)

        buildTimeString = "\nClustered data in "+str(time.time()-buildTimeStart)+" secs.\n"
        resultsString = self.print_both(buildTimeString,resultsString)

        #Save Results and Cleanup
        self.save_results(clusterer_name_short+"_"+"N"+str(num_clusters)+"_S"+str(seed),resultsString, output_directory)

    def save_results(self, classifier, string, output_directory, bif=False):
        try:
            os.mkdir(output_directory)
        except:
            print("Directory Exists, Continuting.\n")
        
        if(bif):
            output_file_path = os.path.join(output_directory,classifier+"_results.bif")
        else:
            output_file_path = os.path.join(output_directory,classifier+"_results.txt")

        try:
            output_file = open(output_file_path,"x")
        except:
            os.remove(output_file_path)
            print("Removed exisiting file\n")
            output_file = open(output_file_path,"x")

        output_file.write(string) 
        output_file.close()
        print("**** Results saved to :"+output_file_path)

    def print_both(self,print_string, resultsString):
        print(print_string)
        resultsString += print_string
        return resultsString

    def filter_data(self,data):
        print("Filtering Data..\n")
        flter = Filter(classname="weka.filters.supervised.attribute.AttributeSelection")
        aseval = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1"])
        assearch = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])
        flter.set_property("evaluator", aseval.jobject)
        flter.set_property("search", assearch.jobject)
        flter.inputformat(data)
        filtered = flter.filter(data)
        return filtered

class cw2_helper():
    def __init__(self, start=True):
        if(start):
            #increased to 4gb for bayes network.
            jvm.start(max_heap_size="8g")

    def cleanup(self):
        jvm.stop()