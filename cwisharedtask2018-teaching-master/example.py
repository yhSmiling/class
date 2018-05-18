from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score
import time

def execute_demo(language,flag):
    data = Dataset(language)

    if flag==0:
        print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))#data.trainset 是dataset函数内返回的dataset的形式  data.devset用来测试用的
    if flag==1:
        print("{}: {} training - {} test".format(language, len(data.trainset), len(data.testset)))
    # for sent in data.trainset:
    #    # print(sent['sentence'], sent['target_word'], sent['gold_label'])
    #    print(sent)

    baseline = Baseline(language)

    baseline.train(data.trainset)

    predictions_devset = baseline.test(data.devset)
    predictions_testset = baseline.test(data.testset)

    gold_labels_devset = [sent['gold_label'] for sent in data.devset]##输出的是二元值  0 1 0 1形式的
    gold_labels_testset =[sent['gold_label'] for sent in data.testset]

    if flag==0: 
        print("Test by using dev set:")
        report_score(gold_labels_devset, predictions_devset)
    if flag==1:
        print("Test by using test set:")
        report_score(gold_labels_testset, predictions_testset)

if __name__ == '__main__':

    flag=0
    start = time.clock()
    execute_demo('english',flag)##将english的数据集导入


    execute_demo('spanish',flag)##先处理英语  再出来西班牙语
    elapsed = (time.clock() - start)
    print("running ",elapsed," seconds")

##############################################################################################
    flag=1
    start1 = time.clock()
    execute_demo('english',flag)

    execute_demo('spanish',flag)##先处理英语  再出来西班牙语
    elapsed1 = (time.clock() - start1)
    print("running ",elapsed1," seconds")










