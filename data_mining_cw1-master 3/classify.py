import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.core.classes import Random
import weka.plot.classifiers as plcls  # NB: matplotlib is required

data_dir = "fer2018/"

try:
    jvm.start()
    
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(data_dir + "fer2018.arff")
    data.class_is_first()

    # print(data)

    cls = Classifier(classname="weka.classifiers.lazy.IBk", options=["-K", "3"])
    cls.build_classifier(data)

    for index, inst in enumerate(data):
        pred = cls.classify_instance(inst)
        dist = cls.distribution_for_instance(inst)
        print(str(index+1) + ": label index=" + str(pred) + ", class distribution=" + str(dist))

    evl = Evaluation(data)
    evl.crossvalidate_model(cls, data, 10, Random(1))

    print(evl.percent_correct)
    print(evl.summary())
    print(evl.class_details())

    plcls.plot_roc(evl, class_index=[0, 1, 2, 3, 4, 5, 6], wait=True)

finally:
    jvm.stop()