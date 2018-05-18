# from sklearn.linear_model import LogisticRegression  #调包
from sklearn import svm
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
# from sklearn.linear_model import LogisticRegression
from sklearn import svm  

class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':#word length feature
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2
        self.model = svm.SVC()#use SVM as classfier
        # self.model = LogisticRegression()

    def extract_features(self, word, whole_sentence,tuple_all_char,dct_sentence,trigram_char):##get the target word feature
        ###################word length####################    
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        ###################word frequency############
        occur_word_count=0 
        for i in whole_sentence: 
            if word in i:
                occur_word_count=occur_word_count+1
        ###################bigram-char frequency############
        record_char_occur_count=0
        record_char_occur_count_trigram=0
        list_char_in_word=[]
        list_for_tuple=[]
        list_trigram_char=[]       
        for k in word:
            list_char_in_word.append(k)
        for i in range(len(list_char_in_word)):
            if i==0:
                tuple_char=('None',list_char_in_word[i])
                trigram_feature_char=('None','None',list_char_in_word[i])
            if i==1:
                tuple_char=(list_char_in_word[i-1],list_char_in_word[i])
                trigram_feature_char=('None',list_char_in_word[i-1],list_char_in_word[i])
            else:
                tuple_char=(list_char_in_word[i-1],list_char_in_word[i])
                trigram_feature_char=(list_char_in_word[i-2],list_char_in_word[i-1],list_char_in_word[i])
            list_for_tuple.append(tuple_char) ##store all bigram-char of target word
            list_trigram_char.append(trigram_feature_char)
        
        for i in list_for_tuple:
            if i in tuple_all_char:
                record_char_occur_count=record_char_occur_count+1
        for i in list_trigram_char:
            if i in trigram_char:
                record_char_occur_count_trigram=record_char_occur_count_trigram+1
        ###########################document frequency#############################
        count_sentence=0
        new_list=[]
        for i in dct_sentence:
            for z,k in i.items():
                if word in k:
                    new_dct={z:k}
                    new_list.append(new_dct)

        for i in new_list:
            for z,l in i.items():
                for p in new_list:
                    for o,m in p.items():
                        if z!=o and l==m:
                            count_sentence=count_sentence+1
        ###########################Synonyms feature#############################
        syn_number=len(wn.synsets(word))

        return [len_chars, len_tokens,occur_word_count,record_char_occur_count,record_char_occur_count_trigram,count_sentence,syn_number]
############################################################################################################################################
    def train(self, trainset): #training data
        X = []
        y = []
        ##############deal with word frequency#########################     
        whole_sentence_train=[]
        dct_all_train_sentence=[]
        for sent in trainset:
            whole_sentence_train.append(sent['sentence'])
            
            dct_for_sentence={sent['hit_id']:sent['sentence']}       
            dct_all_train_sentence.append(dct_for_sentence)

        whole_sentence_train=set(whole_sentence_train)  #获得 所有的句子

        seen=set()
        dct_non_repet_sentence_train=[]
        for d in dct_all_train_sentence:
            t=tuple(d.items())
            if t not in seen:
                seen.add(t)
                dct_non_repet_sentence_train.append(d)

        ###################all bigram for word and char#########################    
        list_tuple_all_char_train_set=[]
        list_for_trigram_char_train_set=[]
        list_tuple_all_word_train_set=[]
        whole_sentence_list=[]
        for i in whole_sentence_train:
            list_single_sentence=[]
            new_sentence=re.sub('[,.!\':""“”’-]','',i)##remove Punctuation
            words=word_tokenize(new_sentence)
            whole_sentence_list.append(words)
            for char in new_sentence:
                list_single_sentence.append(char)
            for j in range(len(list_single_sentence)):
                if j==0:
                    tuple_char=('None',list_single_sentence[j])
                    trigram_char=('None','None',list_single_sentence[j])
                if j==1:
                    trigram_char=('None',list_single_sentence[j-1],list_single_sentence[j])
                    tuple_char=(list_single_sentence[j-1],list_single_sentence[j])
                else:
                    trigram_char=(list_single_sentence[j-2],list_single_sentence[j-1],list_single_sentence[j])
                    tuple_char=(list_single_sentence[j-1],list_single_sentence[j])
                # print(trigram_char)
                list_for_trigram_char_train_set.append(trigram_char)
                list_tuple_all_char_train_set.append(tuple_char)##

        ############################use classifier to make classification###################            
        for sent in trainset:
            X.append(self.extract_features(sent['target_word'],whole_sentence_train,list_tuple_all_char_train_set,dct_non_repet_sentence_train,list_for_trigram_char_train_set)) 
            y.append(sent['gold_label'])
        self.model.fit(X, y)##X is training data  y is the corresponding marker vector   

    def test(self, testset):
        X = []
        whole_sentence_test=[]
        dct_all_test_sentence=[]
        for sent in testset:
            whole_sentence_test.append(sent['sentence'])
            dct_for_sentence={sent['hit_id']:sent['sentence']}       
            dct_all_test_sentence.append(dct_for_sentence)

        whole_sentence_test=set(whole_sentence_test)
        seen=set()
        dct_non_repet_sentence_test=[]
        for d in dct_all_test_sentence:
            t=tuple(d.items())
            if t not in seen:
                seen.add(t)
                dct_non_repet_sentence_test.append(d)


        list_for_trigram_char_test_set=[]
        # list_tuple_all_word_test_set=[]
        list_tuple_all_char_test_set=[]
        for i in whole_sentence_test:
            list_single_sentence=[]
            new_sentence=re.sub('[,.!\':""“”’-]','',i)
            words=word_tokenize(new_sentence)

            for char in new_sentence:
                list_single_sentence.append(char)
            for j in range(len(list_single_sentence)):
                if j==0:
                    tuple_char=('None',list_single_sentence[j])
                    trigram_char=('None','None',list_single_sentence[j])
                if j==1:
                    trigram_char=('None',list_single_sentence[j-1],list_single_sentence[j])
                    tuple_char=(list_single_sentence[j-1],list_single_sentence[j])
                else:
                    trigram_char=(list_single_sentence[j-2],list_single_sentence[j-1],list_single_sentence[j])
                    tuple_char=(list_single_sentence[j-1],list_single_sentence[j])
                list_for_trigram_char_test_set.append(trigram_char)
                list_tuple_all_char_test_set.append(tuple_char)

        for sent in testset:
            X.append(self.extract_features(sent['target_word'],whole_sentence_test,list_tuple_all_char_test_set,dct_non_repet_sentence_test,list_for_trigram_char_test_set))

        return self.model.predict(X) #get prediction












