#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark
import sys
import json
import random
import time
from itertools import combinations

sc=pyspark.SparkContext()
start=time.time()

input_path=sys.argv[1]
outpath=sys.argv[2]
textasrdd=sc.textFile(input_path)
jsontordd=textasrdd.map(lambda x: json.loads(x))
# In[ ]:

def genHashFuncs(num_of_func, baskets):

    func_list = list()

    def build_func(param_a, param_b, param_m):
        def apply_funcs(input_x):
            return ((param_a * input_x + param_b) % 6291469) % param_m

        return apply_funcs

    param_as = random.sample(range(1000000, 90000000), num_of_func)
    param_bs = random.sample(range(1000000, 90000000), num_of_func)
    for a, b in zip(param_as, param_bs):
        func_list.append(build_func(a, b, baskets))

    return func_list

def signature_matrix(hashed_user, function_list,num_of_func):

    signaturematrix = []
    userlist = list(hashed_user)
    for i in range(0, num_of_func):
        signaturematrix.append(float('inf'))

    for user in userlist:
        k=0
        for func in function_list:
            mid = func(user)

            if mid < signaturematrix[k]:
                signaturematrix[k] = mid
            k+=1

    return signaturematrix


# In[ ]:
def bandcreation(chunk, numofbands, numofrows, numofhash):
    b = list(chunk)
    business = b[0]
    sm = b[1]
    bandlist = []
    bandnum = 0
    i = 0
    while i < numofhash:
        band = tuple(sm[i:i + numofrows])
        bandlist.append(((bandnum, band), business))
        bandnum += 1
        i = i + numofrows

    return bandlist


# In[ ]:
def jaccardsimilarity(x, dictionary):
    business1 = x[0]
    business2 = x[1]
    user1 = set(dictionary[business1])
    user2 = set(dictionary[business2])

    i = user1.intersection(user2)
    u = user1.union(user2)

    js = float(len(i) / len(u))

    return js







# In[27]:



#length_users=jsontordd.map(lambda x: x['user_id']).distinct().collect()


# In[93]:
business_dict=jsontordd.map(lambda x:x['business_id']).distinct().zipWithIndex().collectAsMap()
reverse_business_dict={v:k for k,v in  business_dict.items()}

unique_users=jsontordd.map(lambda x:x['user_id']).distinct().zipWithIndex()
users_dict=unique_users.collectAsMap()



# In[ ]:

rdd=jsontordd.map(lambda x: (business_dict.get(x['business_id']),users_dict.get(x['user_id']))).groupByKey().mapValues(lambda x: set(x))
num_of_func=50
length_users=len(users_dict)
function_list=genHashFuncs(num_of_func, length_users*2)
sm=rdd.map(lambda x: (x[0],signature_matrix(x[1],function_list,num_of_func)))
#sm_collection = sm.collect()








numofhash=50
numofbands=50
numofrows=numofhash//numofbands


# In[97]:


b=sm.flatMap(lambda x: bandcreation(x,numofbands,numofrows,numofhash))
#b_collection = b.collect()

            
bb=b.groupByKey().map(lambda x: list(x[1])).filter(lambda x: len(x)>1)
#bb_collection = bb.collect()

# In[ ]:

candidates=bb.flatMap(lambda x: combinations(x,2)).distinct()
#candidates_collection = candidates.collect()



# In[ ]:



bu_temp = rdd.collect()
business_user_dict = {}
for item in bu_temp:
    key = item[0]
    business_user_dict[key] = item[1]  ## set of users

# In[ ]:

jsp=candidates.map(lambda x: (x,jaccardsimilarity(x,business_user_dict))).filter(lambda x: x[1]>=0.05).collect()
with open(outpath,'w') as task1_ans:
    for element in jsp:
        set_list=list(element[0])
        mid_dict={}
        mid_dict['b1']=reverse_business_dict[set_list[0]]
        mid_dict['b2']=reverse_business_dict[set_list[1]]
        mid_dict['sim']=element[1]
        json.dump(mid_dict,task1_ans)
        task1_ans.write('\n')

end = time.time()
print(end - start)

# In[111]:




# In[108]:




# In[ ]:





# In[109]:



    


# In[115]:





# In[133]:




# In[ ]:





# In[134]:



### candidate pairs



# In[140]:




# In[141]:


    



    



# In[142]:





# In[ ]:


#with open(outpath,'w') as task1_ans:
#    for element in jsp:
#        set_list=list(element[0])
#        mid_dict={}
#        mid_dict['b1']=set_list[0]
#        mid_dict['b2']=set_list[1]
#        mid_dict['sim']=element[1]
#        json.dump(mid_dict,task1_ans)
#        task1_ans.write('\n') 

    


# In[ ]:





# In[ ]:




