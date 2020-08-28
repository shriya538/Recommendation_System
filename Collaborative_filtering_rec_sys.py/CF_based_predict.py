#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pyspark
import sys
import json
import random
import time


# In[ ]:


conf = pyspark.SparkConf().setMaster("local").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
sc = pyspark.SparkContext(conf=conf)


# In[ ]:





# In[ ]:


start = time.time()
train_file_path = sys.argv[1]
test_file_path = sys.argv[2]
model_file_path = sys.argv[3]
output_file_path = sys.argv[4]

m_type = sys.argv[5] # either "item_based" or "user_based"
bus_avg_path = "../resource/asnlib/publicdata/business_avg.json"
user_avg_path = "../resource/asnlib/publicdata/user_avg.json"


# In[ ]:


def makepredictionitem(element, data_dict, avg_dict, reverse_dict):
    target = element[0]
    target_str = reverse_dict.get(target, "None")
    business_score_list = list(element[1])
    result = list()
    for item in business_score_list:
        if target < item[0]:
            key = tuple((target, item[0]))
        else:
            key = tuple((item[0], target))

        result.append(tuple((item[1], data_dict.get(key, 0))))

    score_sim_list = sorted(result, key=lambda item: item[1], reverse=True)[:3]
    numerator = sum(map(lambda item: item[0] * item[1], score_sim_list))
    if numerator == 0:
        return tuple((target, avg_dict.get(target_str, 3.823989)))
    denominator = sum(map(lambda item: abs(item[1]), score_sim_list))
    if denominator == 0:
        return tuple((target, avg_dict.get(target_str, 3.823989)))

    ans = tuple((target, numerator / denominator))
    return ans


def makepredictionuser(element, reverse_dict, avg_dict, data_dict):
    target = element[0]
    target_uid_str = reverse_dict.get(target, "None")
    uids_score_list = list(element[1])  # list of tuple(uidx, score)
    result = list()
    for uids_score in uids_score_list:
        if target < uids_score[0]:
            key = tuple((target, uids_score[0]))
        else:
            key = tuple((uids_score[0], target))

        other_user_str = reverse_dict.get(uids_score[0], "None")
        avg_score = avg_dict.get(other_user_str, 3.823989)
        result.append(tuple((uids_score[1], avg_score, data_dict.get(key, 0))))

    numerator = sum(map(lambda x: (x[0] - x[1]) * x[2], result))
    if numerator == 0:
        return tuple((target, avg_dict.get(target_uid_str, 3.823989)))
    denominator = sum(map(lambda item: abs(item[2]), result))
    if denominator == 0:
        return tuple((target, avg_dict.get(target_uid_str, 3.823989)))

    ans = tuple((target, avg_dict.get(target_uid_str, 3.823989) + (numerator / denominator)))
    return ans


data = sc.textFile(train_file_path)
jsontordd = data.map(lambda x: json.loads(x))
train_load = jsontordd.map(lambda x: (x['user_id'], x['business_id'], x['stars']))

# In[ ]:


user_dict = train_load.map(lambda x: x[0]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()

# In[ ]:


reverse_user_dict = {}
for k, v in user_dict.items():
    reverse_user_dict[v] = k

# In[ ]:
business_dict = train_load.map(lambda x: x[1]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()

reverse_business_dict = {}
for k, v in business_dict.items():
    reverse_business_dict[v] = k

test_load = sc.textFile(test_file_path).map(lambda x: json.loads(x))

# In[ ]:
if m_type == "user_based":
    user_avg = sc.textFile('../resource/asnlib/publicdata/user_avg.json').map(lambda row: json.loads(row)).map(lambda x: dict(x)).flatMap(
        lambda y: y.items()).collectAsMap()

    model_sim_dict = sc.textFile(model_file_path).map(lambda x: json.loads(x)).map(
        lambda x: {(user_dict.get(x['u1']), user_dict.get(x['u2'])): x['sim']}).flatMap(lambda x: x.items()).collectAsMap()


    def train_func(y):
        result = []
        for item in list(set(y)):
            t = [item[0], item[1]]
            result.append(tuple(t))
        return result


    train_user_business = train_load.map(lambda x: (business_dict[x[1]], (user_dict[x[0]], x[2]))).groupByKey().map(
        lambda x: (x[0], train_func(x[1])))

    test_user_business = test_load.map(
        lambda x: (business_dict.get(x['business_id'], -1), user_dict.get(x['user_id'], -1))).filter(
        lambda x: x[0] != -1 and x[1] != -1)

    output = test_user_business.leftOuterJoin(train_user_business).mapValues(
        lambda x: makepredictionuser(tuple(x), reverse_user_dict, user_avg, model_sim_dict))

    final = output.map(
        lambda x: {"user_id": reverse_user_dict.get(x[1][0]), "business_dict": reverse_business_dict.get(x[0]),
                   "stars": x[1][1]})
    final_collect = final.collect()
    with open(output_file_path, 'w') as outpath:
        for item in final_collect:
            outpath.write(json.dumps(item))
            outpath.write('\n')
        outpath.close()
    end=time.time()
    print(end-start)



else:
    if m_type == 'item_based':
        bus_avg_dict = sc.textFile(bus_avg_path).map(lambda row: json.loads(row)).map(lambda x: dict(x)).flatMap(lambda y: y.items()).collectAsMap()

        model_sim_dict = sc.textFile(model_file_path).map(lambda x: json.loads(x))
        model_sim_dict_count=model_sim_dict.count()
        model_sim_dict=model_sim_dict. map(lambda x: {(business_dict.get(x['b1']), business_dict.get(x['b2'])): (x['sim'])})
        model_sim_dict_count=model_sim_dict.count()
        model_sim_dict=model_sim_dict.flatMap(lambda x: x.items()).collectAsMap()


        # In[ ]:

        def train_func(y):
            result = []
            for item in list(set(y)):
                t = [item[0], item[1]]
                result.append(tuple(t))
            return result


        train_user_business = train_load.map(lambda x: (user_dict[x[0]], (business_dict[x[1]], x[2]))).groupByKey().map(
            lambda x: (x[0], train_func(x[1])))

        test_user_business = test_load.map(
            lambda x: (user_dict.get(x['user_id'], -1), business_dict.get(x['business_id'], -1))).filter(
            lambda x: x[0] != -1 and x[1] != -1)

        output = test_user_business.leftOuterJoin(train_user_business).mapValues(
            lambda x: makepredictionitem(tuple(x), model_sim_dict, bus_avg_dict, reverse_business_dict))

        final = output.map(lambda x: {"user_id": reverse_user_dict.get(x[0]), "business_dict": reverse_business_dict.get(x[1][0]), "stars": x[1][1]})
        final_collect = final.collect()

        with open(output_file_path, 'w') as outpath:
            for item in final_collect:
                outpath.write(json.dumps(item)+'\n')

        outpath.close()
        end=time.time()
        print(end-start)
        print('hey')


# In[ ]:




# In[ ]:




