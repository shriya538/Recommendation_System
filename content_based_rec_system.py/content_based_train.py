#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys 
import pyspark
import time
import json 

conf = pyspark.SparkConf().setMaster("local").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
sc = pyspark.SparkContext(conf=conf)
input_path=sys.argv[1]


data=sc.textFile(input_path)

jsontordd=data.map(lambda x: json.loads(x))


# In[16]:


model_path=sys.argv[2]
##start preprocessing
N=jsontordd.map(lambda x:(x['business_id'],1)).groupByKey().count()
print(N)
pp=jsontordd.map(lambda x: (x['business_id'],x['text'])).reduceByKey(lambda a,b: a+b) ## business merge text to make only unique buz and their text


# In[3]:


import re
import string
def punctuation(chunk):
    # x is a list of words wanna remove punctuation 
    new=[]
    for word in chunk:
        temp=re.split('[\)\[,.$!?:%0123456789;"\]\(]',word)
        new.append(temp)
    
    final=[]
    for line in new:
        for i in line:
            if i!='':
                final.append(str(i))
    
    return final

def stop(chunk,sw):
    new=[]
    for word in chunk:
        if word not in sw and len(word)>2:
            new.append(word)
            
    return new

stopwords_path=sys.argv[3]
stopwords=sc.textFile(stopwords_path)
sw=stopwords.flatMap(lambda x: x.split()).collect()
    
#lower,split,punctuation,sw
##pp2=pp.map(lambda x:( x[0],x[1].lower())).map(lambda x: (x[0],x[1].split())).map(lambda x:(x[0],punctuation(x[1]))).map(lambda x:(x[0],stop(x[1],sw))).take(1)
##print(pp2)
mapr1=pp.map(lambda x:(x[0],x[1].lower())).flatMapValues(lambda x: re.split(r'['+string.whitespace + string.punctuation+']',x)).mapValues(lambda x: x.strip(r'['+string.punctuation+ string.digits+']')).filter(lambda x: x[1] not in sw and x[1]!='').map(lambda x:((x[0],x[1]),1)).reduceByKey(lambda a,b:a+b)
                                               
                                               
                                       
#(businessid,word),count


#pp2=pp.flatMapValues(lambda x: x.split()).map(lambda x: (x[0],x[1].lower())).flatMapValues(lambda x: re.split('[\)\[,.$!?:%0123456789;"\]\(]',x)).filter(lambda x: x[1]!='')
#split,lowered,punctuation removed,''
#pp2=pp.map(lambda x:(x[0],x[1].lower())).flatMapValues(lambda x: re.split('[\)\[,.$!?:%0123456789;"\]\(]',x)).groupByKey()

##All preprocessing done till pp2 lower,split,punctuation,sw


# In[4]:


def func(x):
    maxval=0
    i=1
    for word in x:
        w=list(word)
        for item in word:
            if i==2:
                if maxval<item:
                    maxval=item
            i+=1
        i=1
            
    return maxval

mapr2=mapr1.map(lambda x:(x[0][0],(x[0][1],x[1]))).groupByKey().map(lambda x:((x[0],func(x[1])),(list (x[1])))).flatMap(lambda x:[((val[0],x[0][0]),(val[1],x[0][1])) for val in x[1]])

#(businessid),[(word,count),(word,count)...]
#func x finds the max freq word in the document
#[(('paid', 'bZMcorDrciRbjdjRyANcjA'), (8, 16)), (('store', 'bZMcorDrciRbjdjRyANcjA'), (2, 16))]


# In[5]:



import math    
mapr3=mapr2.map(lambda x:((x[0][0]),(x[0][1],x[1][0],x[1][1]))).groupByKey().mapValues(lambda x: (list(x),len(x)))

# [('store', ([('bZMcorDrciRbjdjRyANcjA', 2, 16),('WLu8QHuN6zjlBEJ7HXIg1Q', 3, 7)], 4480))] eg full


def calculate_tfidf(x,N):
   chunk=list(x)
   second=chunk[1]
   num=second[1]
   final=list()
   #print('Deathnote')
   for val in second[0]:
       
       temp=list(val)
       #print(temp[0])
       tf=temp[1]/temp[2]
       #print(tf)
       #print(num)
       idf=math.log(N/num)
       #print('idf')
       #print(idf)
       ans=tf*idf
       #print(ans)
       entry=[chunk[0],temp[0],ans]
       #print('\n\n\n')
       final.append(entry)
       
   return final
    

N=jsontordd.map(lambda x:(x['business_id'],1)).distinct().count() 
## No of documents
tfidf=mapr3.map(lambda x: calculate_tfidf(x,N)).flatMap(lambda x:x).map(lambda x: (x[1],(x[0],x[2]))).groupByKey().mapValues(list)
## [['store', ([('bZMcorDrciRbjdjRyANcjA', 2, 16),('WLu8QHuN6zjlBEJ7HXIg1Q', 3, 7)], 4480)]] eg
##[['store', 'bZMcorDrciRbjdjRyANcjA', 0.10483992453037144], ['store', 'n8Zqqhff-2cxzWt_nwhU2Q', 0.0066741596518008345], ['store', 'VfFHPsPtTW4Mgx0eHDyJiQ', 0.014460679245568474], ['store', 'wqFkAsxYPA5tcdSkYMtrrw', 0.0535352806112535]]
   


# In[6]:


## for each business you want top 200 words with highest tfidf
## (business,[(word1,score),(word2, score).....])
import json
def self_top(x):
    ans=[]
    i=0
    for word in x:
        ans.append(word[0])
        if i==199:
            break
        i+=1
    return ans
      

## business dictionary
bus_dict=jsontordd.map(lambda x:x['business_id']).distinct().sortBy(lambda x:x).zipWithIndex().collectAsMap()
## 
#words_dict=business_profile.flatMap(lambda x: x[1]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()

business_profile_temp=tfidf.map(lambda x:(bus_dict[x[0]],sorted(x[1],key=lambda x:-1*(x[1])))).mapValues(lambda x: self_top(x))

words_dict=business_profile_temp.flatMap(lambda x: x[1]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()

def word_to_num(x,words_dict):
    ans=[]
    for word in x:
        ans.append(words_dict[word])
    return ans
        

business_profile=business_profile_temp.mapValues(lambda x: word_to_num(x,words_dict))

bp=business_profile.collectAsMap()

business_prof_list=[]
for k,v in bp.items():
    temp={}
    temp["type"]="business_profile"
    temp["business_id"]=k
    temp["business_profile"]=v
    business_prof_list.append(temp)





#[('bZMcorDrciRbjdjRyANcjA', [('tanning', 4.490393496306841), ('beds', 2.0074237698181614), ('corporate', 1.8438074180207324)..]),()]
#['business_id',[(tanning),()]]    

#[('6372', ['tanning', 'beds', 'corporate', 'membership', 'tan', 'tara', 'lotion', 'tanned', 'spray', 'salon', 'account', 'month', 'cash', 'maria', 'body', 'sessions', 'monthly', 'tans', 'package', 'maryland', 'girl', 'salons', 'hung', 'bed', 'sagion', 'mrinternational', 'meagain', 'levelhaving', 'bulbs', 'desk', 'pre-summer', 'education', 'heat', 'fgt', 'unquote', "'fine'", 'irrationally', 'pkwy', 'nonchalance', 'advised', 'billing', 'bodyheat', 'hmmmmmmm', 'signed', 'credit', 'emails', 'unpaid', 'nicei', 'office', 'charges', 'cancelation', 'drop-in', 'aspen', 'tools', 'mari', 'streaky', "maria's", 'mocked', 'accessing', 'adjust', 'paid', 'baffles', 'tooi', 'directed', 'provide', 'authorize', 'trust', 'extreme', 'fee', 'owes', 'blonde', 'cancel', 'hp', 'unacceptable', 'phones', 'bureau', 'upgraded', 'girls', 'charged', 'interested', 'retailer', 'damaging', 'streaks', 'card', 'voice', 'honoring', 'unused', 'company', 'authorized', 'straightened', 'rays', 'lotions', 'dispute', 'disabled', 'medium', 'ensures', 'pressure', 'institution', 'owed', 'lvac', 'trial', 'email', 'months', 'versa', 'convenience', 'reps', 'clear', 'loli', 'machines', 'sun', 'horizon', 'silverado', 'locations', 'financial', 'desires', 'rudeness', 'argument', 'voicemail', 'told', 'response', 'thingy', 'stolen', 'canceled', 'rude', 'reverse', 'customer', 'tax', '-hour', 'sign', 'slipped', 'owe', 'contact', 'disrespectful', 'agreed', 'manager', 'atm', 'raising', 'parkway', 'scan', 'physically', 'figuring', 'called', 'location', 'deals', 'scare', 'palm', 'uneven', 'pho', 'ummm', 'requesting', 'verdict', 'clueless', 'scam', 'downright', 'partial', 'convince', 'remembers', 'file', 'pays', 'hired', 'leading', 'spoke', 'angel', 'countless', 'machine', 'line', 'shade', 'significantly', 'promotion', 'call', 'directions', 'attempted', 'shady', 'days', 'direct', 'workers', 'session', 'ultimately', 'record', 'beach', 'staring', 'client', 'automatically', 'responded', 'mail', 'fees', 'stops', 'purchasing', 'contacted', 'spa', 'depends', 'opposed', 'insisted', 'continues', 'listened', 'reviewed', 'city', 'bang', 'level', 'eastern'])]
###BUSINESS PROFILE DONE
#[(6372, [112321, 10058, 24995, 71173, 112232, 112445, 67034, 112317, 107146, 69435, 98072, 697, 73837, 18048, 101106, 12744, 73846, 112331, 81847, 69771, 46624, 98083, 54456, 10014, 97810, 74592, 70563, 64964, 15359, 30046, 35100, 51420, 40729, 57844, 86000, 120438, 77441, 1490, 11459, 12754, 52844, 103080, 26060, 120355, 35964, 78954, 19401, 17019, 6222, 76811, 116357, 109144, 69434, 73281, 586, 1247, 81986, 8408, 31186, 89657, 7071, 118019, 39053, 116332, 81650, 40339, 12349, 17017, 119252, 54063, 85001, 15637, 120850, 46647, 19392, 57278, 94925, 27716, 109143, 17557, 53410, 123271, 120734, 7072, 23271, 109031, 67038, 31878, 31255, 36783, 70845, 57102, 88537, 81642, 67784, 117607, 35959, 73848, 24491, 122477, 94341, 21827, 68145, 66185, 53620, 103195, 41066, 30039, 97223, 5618, 123274, 116081, 94655, 66458, 110285, 108806, 97217, 17021, 114567, 95251, 91984, 27199, 104197, 103065, 81641, 112815, 24262, 31897, 2077, 91461, 82764, 6564, 68870, 85122, 99201, 40904, 16777, 28415, 99238, 66149, 119859, 119214, 84968, 94383, 122362, 22207, 82173, 33222, 99182, 82900, 24592, 93914, 110218, 40933, 52718, 106955, 64358, 83489, 25383, 4192, 68138, 101345, 103087, 65536, 33753, 89372, 31192, 16762, 6693, 101362, 28206, 31185, 101103, 119138, 92845, 9748, 107944, 7117, 94648, 108901, 40393, 21927, 90221, 24263, 29728, 80165, 56975, 24370, 51069, 106239, 65743, 95271, 126867, 21478, 68460, 8848, 91103, 64960, 6394, 22778])]    


# In[ ]:





# In[11]:



def convert_to_one_list(x1,x2):
    result=list(x1)
    result.extend(x2)
    return result
    
    
    return ans
            

def business_to_num(x,bus_dict):
    ans=[]
    for business in x:
        ans.append(bus_dict[business])
    return ans
        
def bid_to_words_num(x,bp):
    ans=[]
    for bid in x:
        ans.append(bp[bid])
    return ans
    
    
##User Profile done
user_dict=jsontordd.map(lambda x:x['user_id']).distinct().sortBy(lambda x:x).zipWithIndex().collectAsMap()

user_prof1=jsontordd.map(lambda x:(x['user_id'],x['business_id'])).groupByKey().map(lambda x: (user_dict[x[0]],list(set(x[1])))).mapValues(lambda x:business_to_num(x,bus_dict)).flatMapValues(lambda x: bid_to_words_num(x,bp)).reduceByKey(convert_to_one_list).filter(lambda x: len(x[1])>1)
user_prof=user_prof1.collectAsMap()

#map(lambda x:{x[0]:list(x[1])})

#joinmaybe=user.join(business_profile).map(lambda x: (user_dict[x[1][0]],x[1][1])).groupByKey().collect()
#print(joinmaybe)





## error while collecting
#for k,v in joinmaybe.items():
#    temp={}
#    temp["type"]="user_profile"
#    temp["user_id"]=k
#    temp["user_profile"]=v
#    user_prof.append(temp)



    

#print(user_profile)


# In[22]:


#add bus_dict,user_dict,business_profile,user_profile

## user_profile
def d_to_model(data, t, k1,k2):
    result = list()
    for key, val in data.items():
        temp={}
        temp['type']=t
        temp[k1]=key
        temp[k2]=val
        result.append(temp)
        
    return result

def l_to_model(data,t,k1,k2):
    result=list()
    for element in data:
        for k,v in element.items():
            temp={}
            temp['type']=t
            temp[k1]=k
            temp[k2]=v
            result.append(temp)
    return result


model=[]
model.extend(d_to_model(bus_dict,"business_dict","business_id","business_index"))
model.extend(d_to_model(user_dict,"user_dict","user_id","user_index"))
model.extend(business_prof_list)
model.extend(d_to_model(user_prof,"user_profile","user_index","user_profile"))






# In[24]:


with open(model_path,'w') as outpath:
    for item in model:
        outpath.writelines(json.dumps(item)+"\n")
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




