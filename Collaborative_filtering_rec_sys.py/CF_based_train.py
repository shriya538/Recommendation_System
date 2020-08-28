#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pyspark
import sys
import json
import random
import time
import math
from itertools import combinations
from collections import defaultdict

sc=pyspark.SparkContext()
sc.setLogLevel("ERROR")
start=time.time()


# In[3]:


input_path= sys.argv[1]
outpath=sys.argv[2]



# In[4]:


mtype=sys.argv[3] ## or can be item based


# In[19]:


data=sc.textFile(input_path)
jsontordd=data.map(lambda x: json.loads(x))
rdd=jsontordd.map(lambda x:(x['user_id'],x['business_id'],x['stars']))


# In[31]:


# user_dict
user_dict= rdd.map(lambda x: x[0]).distinct().sortBy(lambda x:x).zipWithIndex().collectAsMap()

#user_dict=dict(zip(user_collect,range(1,len(user_collect)+1)))
reverse_user_dict= {}
for k,v in user_dict.items():
    reverse_user_dict[v]=k




# In[40]:




# In[43]:


# business_dict
business_collect= rdd.map(lambda x: x[1]).distinct().sortBy(lambda x:x).collect()
business_dict=dict(zip(business_collect,range(0,len(business_collect)+1)))
reverse_business_dict={}
for k,v in business_dict.items():
    reverse_business_dict[v]=k




# In[51]:





# In[53]:


def genHashFuncs(num_of_func, baskets):

    func_list = list()

    def build_func(param_a, param_b, param_m):
        def apply_funcs(input_x):
            return ((param_a * input_x + param_b) % 2333333) % param_m

        return apply_funcs

    param_as = random.sample(range(1, 90000000), num_of_func)
    param_bs = random.sample(range(0, 90000000), num_of_func)
    for a, b in zip(param_as, param_bs):
        func_list.append(build_func(a, b, baskets))

    return func_list

def signature_matrix(hashed_business, function_list,num_of_func):

    signaturematrix = []
    businesslist = list(hashed_business)
    for i in range(0, num_of_func):
        signaturematrix.append(float('inf'))

    for business in businesslist:
        k=0
        for func in function_list:
            mid = func(business)

            if mid < signaturematrix[k]:
                signaturematrix[k] = mid
            k+=1

    return signaturematrix


# In[78]:


def bandcreation(chunk, numofbands, numofrows, numofhash):
    b = list(chunk)
    user = b[0]
    sm = b[1]
    bandlist = []
    bandnum = 0
    i = 0
    while i < numofhash:
        band = tuple(sm[i:i + numofrows])
        bandlist.append(((bandnum, band), user))
        bandnum += 1
        i = i + numofrows

    return bandlist


# In[ ]:


def jaccardsimilarity(x, dictionary):
    user1 = x[0]
    user2 = x[1]
    business1 = set(dictionary[user1])
    business2 = set(dictionary[user2])
    
    if len(business1 & business2)>=3:

        i = business1.intersection(business2)
        u = business1.union(business2)

        js = float(len(i) / len(u))
        if js>=0.01:
            return True

    return False

def existrecords(dict1, dict2):

    if len(dict1) !=0 and len(dict2)!=0 :
        if len(set(dict1.keys()) & set(dict2.keys())) >= 3:
            return True

        else:
            return False
    return False



def pearson_similarity(dict1,dict2):
    co_rated_user= list((set(dict1.keys())).intersection(dict2.keys()))
    stars_dict1=[]
    stars_dict2=[]
    
    for user in co_rated_user:
        stars_dict1.append(dict1[user])
        stars_dict2.append(dict2[user])
    
    if len(stars_dict1)!=0 and len(stars_dict2)!=0:
        avg1=sum(stars_dict1)/len(stars_dict1)
        avg2=sum(stars_dict1)/len(stars_dict2)

        numerator=0
        for i in range(len(stars_dict1)):
            term1=stars_dict1[i]-avg1
            term2=stars_dict2[i]-avg2
            product=term1*term2
            numerator+=product
        if numerator ==0:
            return 0

        sum1=0
        sum2=0

        for i in range(len(stars_dict1)):
            term1=(stars_dict1[i]-avg1)**2
            term2=(stars_dict2[i]-avg2)**2
            sum1+=term1
            sum2+=term2
        sqrt1=math.sqrt(sum1)
        sqrt2=math.sqrt(sum2)

        denominator=sqrt1*sqrt2

        if denominator==0:
            return 0
        return numerator/denominator
    
    else:
        return 0

# In[79]:

if mtype=='user_based':
    ##USER BASED
    user_based= rdd.map(lambda x: (user_dict.get(x[0]),(business_dict.get(x[1]),x[2]))).groupByKey().mapValues(list).filter(lambda x:len(x[1])>=3)
    u_based=jsontordd.map(lambda x: (user_dict.get(x['user_id']),business_dict.get(x['business_id']))).groupByKey().mapValues(list).filter(lambda x: len(x[1])>=3)
    
    

    # In[80]:


    # signature matrix construction
    num_of_func=50
    length_business=len(business_dict)
    function_list=genHashFuncs(num_of_func,length_business*2)
    sm=u_based.map(lambda x:(x[0],(signature_matrix(x[1],function_list,num_of_func))))
    

    # In[81]:


    numofhash=50
    numofbands=50
    numofrows=numofhash//numofbands


    # In[85]:


    b=sm.flatMap(lambda x: bandcreation(x,numofbands,numofrows,numofhash))
    bb=b.groupByKey().map(lambda x: list(x[1])).filter(lambda x: len(x)>1)
    

    # In[87]:



    candidates=bb.flatMap(lambda x: combinations(x,2)).distinct()

    


    # In[88]:

    #rdd(userid,bid,stars)
    #user_based - user [business,rating]

    bu_temp = user_based.collect()
    user_business_dict = {}
    for item in bu_temp:
        key = item[0]
        user_business_dict[key]=[]
        for pair in item[1]:
              user_business_dict[key].append(pair[0])## set of business



    # In[ ]:

 

    ## Now you need to calculate pearson


    # In[ ]:
    jsp=candidates.filter(lambda x: jaccardsimilarity(x,user_business_dict))
    
    # dictionary for {'uid':{'bid1':star1,'bid2':star2....}}
    # user_based=[(uid1,[(bid1,star1),(bid2,star2)...])]

    ub_dict=user_based.mapValues(lambda x:[{k:v}for k,v in x]).mapValues( lambda x: {user:score for item in x for user,score in item.items()}).map(lambda x: {x[0]: x[1]}).flatMap(lambda x: x.items()).collectAsMap()

    pearson=candidates.map(lambda x: (x,pearson_similarity(ub_dict.get(x[0]),ub_dict.get(x[1]))))
    
    pearson_filter=pearson.filter(lambda x: x[1]>0)
  
   
    output= pearson_filter.map(lambda x: {
        'u1': reverse_user_dict[x[0][0]],
        'u2': reverse_user_dict[x[0][1]],
        'sim': x[1]
    }).collect()

    with open(outpath,'w') as outputfile:
        for item in output:
            outputfile.writelines(json.dumps(item))
            outputfile.write('\n')
    end=time.time()
    print(end-start)
    
    

elif mtype=='item_based':

    shrunk1=rdd.map(lambda x:(business_dict[x[1]],(user_dict[x[0]],x[2]))).groupByKey()
    shrunk2=shrunk1.mapValues(lambda x: list(x)).filter(lambda x: len(x[1])>=3)
    shrunk3=shrunk2.mapValues(lambda x: [{uid_score[0]: uid_score[1]} for uid_score in x]).mapValues( lambda x: {user:score for item in x for user,score in item.items()})
    

    candidatei=shrunk3.map(lambda x: x[0])

    bid_user_dict=shrunk3.map(lambda x: {x[0]:x[1]}).flatMap(lambda x: x.items()).collectAsMap()

    candidate_pair=candidatei.cartesian(candidatei).filter(lambda x: x[0]<x[1]).filter(lambda x: existrecords(bid_user_dict.get(x[0]),bid_user_dict.get(x[1]))).map(lambda x:(x,pearson_similarity(bid_user_dict.get(x[0]),bid_user_dict.get(x[1])))).filter(lambda x: x[1]>0)

    finalans=candidate_pair.map(lambda x: {'b1':reverse_business_dict[x[0][0]],
                                           'b2':reverse_business_dict[x[0][1]],
                                           'sim':x[1]
                                           }).collect()

    with open(outpath, 'w') as outputfile:
        for item in finalans:
            outputfile.writelines(json.dumps(item))
            outputfile.write('\n')
    end=time.time()
    print(end-start)






















# In[ ]:



