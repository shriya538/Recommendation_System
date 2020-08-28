#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import json
import sys
import pyspark
conf = pyspark.SparkConf().setMaster("local").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
sc = pyspark.SparkContext(conf=conf)


# In[2]:


def cosine_similarity(user_profile,business_profile):
    if len(user_profile)>0 and len(business_profile)>0:
        u_prof=set(user_profile)
        b_prof=set(business_profile)
        numerator=len(u_prof.intersection(b_prof))
        denominator=math.sqrt(len(u_prof))*math.sqrt(len(b_prof))
        cs=numerator/denominator
        return cs
    else:
        return 0.0


# In[27]:


test_file_path=sys.argv[1]


# In[21]:



model_file_path=sys.argv[2]
output_path=sys.argv[3]


modelrdd=sc.textFile(model_file_path).map(lambda x: json.loads(x))
user_index_dict= modelrdd.filter(lambda x: x['type'] == 'user_dict').map(lambda x: {x['user_id']: x['user_index']}).flatMap(lambda x: x.items()).collectAsMap()
reverse_user_index_dict= {v: k for k, v in user_index_dict.items()}

business_index_dict= modelrdd.filter(lambda x: x['type'] == 'business_dict').map(lambda x: {x['business_id']: x['business_index']}).flatMap(lambda x: x.items()).collectAsMap()

reverse_business_index_dict= {v: k for k, v in business_index_dict.items()}

user_profile=modelrdd.filter(lambda x: x['type']=='user_profile').map(lambda x: {x['user_index']:x['user_profile']}).flatMap(lambda x: x.items()).collectAsMap()
business_profile=modelrdd.filter(lambda x: x['type']=='business_profile').map(lambda x: {x['business_id']:x['business_profile']}).flatMap(lambda x: x.items()).collectAsMap()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




# In[22]:


##make prediction 

predict_result1=sc.textFile(test_file_path).map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], x['business_id'])).map(lambda x:(user_index_dict.get(x[0],-100),business_index_dict.get(x[1],-100)))
                                                                              


# In[23]:





# In[25]:


predict_result2=predict_result1.filter(lambda x: x[0]!=-100 and x[1]!=-100).map(lambda x: ((x[0], x[1]), cosine_similarity(user_profile.get(x[0],[]),business_profile.get(x[1],[])))).filter(lambda x:x[1]>=0.01)


# In[ ]:





# In[32]:


predict_result3=predict_result2.map(lambda x: {"user_id":reverse_user_index_dict[x[0][0]],"business_id": reverse_business_index_dict[x[0][1]],"sim": x[1]})


# In[33]:


predict_result4=predict_result3.collect()


# In[ ]:





# In[ ]:





# In[34]:


with open(output_path,'w') as outpath:
    for item in predict_result4:
        outpath.writelines(json.dumps(item)+"\n")
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




