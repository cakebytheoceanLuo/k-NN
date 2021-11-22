# https://medium.com/@kumon/how-to-realize-similarity-search-with-elasticsearch-3dd5641b9adb
# https://docs.aws.amazon.com/opensearch-service/latest/developerguide/knn.html
import sys
import requests
import h5py
import numpy as np
import json
import aiohttp
import asyncio
import time
import httpx
from requests.auth import HTTPBasicAuth
from statistics import mean

# if len(sys.argv) != 2:
#     print("Type in the efSearch!")
#     sys.exit()

# path = '/tmp/sift-128-euclidean.hdf5.1M' # float dataset
# path = '/tmp/sift-128-euclidean.hdf5' # float dataset
path = '/home/ubuntu/sift-128-euclidean.hdf5' # float dataset
output_csv = '/tmp/sift-es.csv'

# url = 'http://127.0.0.1:9200/sift-index/'
host = 'https://vpc-....ap-southeast-1.es.amazonaws.com/' # single node
# host = 'https://vpc-....ap-southeast-1.es.amazonaws.com/' # two nodes
url = host + 'sift-index/'
requestHeaders = {'content-type': 'application/json'} # https://stackoverflow.com/questions/51378099/content-type-header-not-supported
auth = HTTPBasicAuth('admin', 'I#vu7bTAHB')

# Build an index

#https://stackoverflow.com/questions/17301938/making-a-request-to-a-restful-api-using-python
# PUT sift-index
data = '''{
  "settings": {
    "index": {
      "knn": true,
      "knn.space_type": "l2",
      "knn.algo_param.m": 6,
      "knn.algo_param.ef_construction": 50,
      "knn.algo_param.ef_search": 50,
      "refresh_interval": -1,
      "translog.flush_threshold_size": "10gb",
      "number_of_replicas": 0
    }
  },
  "mappings": {
    "properties": {
      "sift_vector": {
        "type": "knn_vector",
        "dimension": 128
      }
    }
  }
}'''
# https://medium.com/@kumon/how-to-realize-similarity-search-with-elasticsearch-3dd5641b9adb

response = requests.put(url, data=data, headers=requestHeaders, auth=HTTPBasicAuth('admin', 'I#vu7bTAHB'))
# response = requests.put(url, data=data, verify=False, headers=requestHeaders, auth=auth)
assert response.status_code==requests.codes.ok

# cluster_url = 'http://127.0.0.1:9200/_cluster/settings'
cluster_url = host + '_cluster/settings'
cluster_data = '''{
  "persistent" : {
    "knn.algo_param.index_thread_qty": 16
  }
}
'''
response = requests.put(cluster_url, data=cluster_data, auth=HTTPBasicAuth('admin', 'I#vu7bTAHB'), headers=requestHeaders)
assert response.status_code==requests.codes.ok

# Bulkload into index
bulk_template = '{ "index": { "_index": "sift-index", "_id": "%s" } }\n{ "sift_vector": [%s] }\n'

hf = h5py.File(path, 'r')

for key in hf.keys():
    print("A key of hf is %s" % key) #Names of the groups in HDF5 file.

vectors = np.array(hf["train"][:])

num_vectors, dim = vectors.shape
print("num_vectors: %d" % num_vectors)
print("dim: %d" % dim)

bulk_data = ""
start = time.time()
for (id,vector) in enumerate(vectors):
  assert len(vector)==dim
  vector_str = ""
  for num in vector:
      vector_str += str(num) + ','
  vector_str = vector_str[:-1]
  id_str = str(id)
  single_bulk_done = bulk_template % (id_str, vector_str)
  bulk_data += single_bulk_done
  if (id+1) % 100000 == 0:
    print(str(id+1))
    # POST _bulk
    response = requests.put(url + '_bulk', data=bulk_data, auth=HTTPBasicAuth('admin', 'I#vu7bTAHB'), headers=requestHeaders)
    assert response.status_code==requests.codes.ok
    bulk_data = ""

end = time.time()
print("Insert Time: %d mins" % ((end - start) / 60.0)) # Unit: min

# refresh_url = 'http://127.0.0.1:9200/sift-index/_settings'
refresh_url = host + 'sift-index/_settings'
refresh_data = '''{
  "index" : {
      "refresh_interval": "1s"
  }
}
'''
response = requests.put(refresh_url, data=refresh_data, headers=requestHeaders, auth=HTTPBasicAuth('admin', 'I#vu7bTAHB'))
assert response.status_code==requests.codes.ok

# response = requests.post('http://127.0.0.1:9200/sift-index/_refresh', verify=False, headers=requestHeaders)
# assert response.status_code==requests.codes.ok

# merge_url = 'http://127.0.0.1:9200/sift-index/_forcemerge?max_num_segments=1'
merge_url = host + 'sift-index/_forcemerge?max_num_segments=1'
merge_response = requests.post(merge_url, headers=requestHeaders, auth=HTTPBasicAuth('admin', 'I#vu7bTAHB'), timeout=600)
assert merge_response.status_code==requests.codes.ok

# warmup_url = 'http://127.0.0.1:9200/_opendistro/_knn/warmup/sift-index'
warmup_url = host + '_opendistro/_knn/warmup/sift-index'
warmup_response = requests.get(warmup_url, headers=requestHeaders, auth=HTTPBasicAuth('admin', 'I#vu7bTAHB'))
assert warmup_response.status_code==requests.codes.ok




# Send queries
total_time = 0 # in ms
hits = 0 # for recall calculation

query_template = '''
{
  "size": 50,
  "query": {"knn": {"sift_vector": {"vector": [%s],"k": 50}}}
}
'''

queries = np.array(hf["test"][:])
nq = len(queries)
neighbors = np.array(hf["neighbors"][:])
# distances = np.array(hf["distances"][:])

num_queries, q_dim = queries.shape
print("num_queries: %d" % num_queries)
print("q_dim: %d" % q_dim)
assert q_dim==dim
ef_search_list = [50, 100, 150, 200, 250, 300]

for ef_search in ef_search_list:
  ef_data = '''{
    "index": {
      "knn.algo_param.ef_search": %d
    } 
  }'''
  ef_data = ef_data % ef_search
  ### Update Index Setting: efSearch
  response = requests.put(url + '_settings', data=ef_data, headers=requestHeaders, auth=HTTPBasicAuth('admin', 'I#vu7bTAHB'))
  assert response.status_code==requests.codes.ok
  total_time_list = []
  hits_list = []
  for count in range(5):
    total_time = 0 # in ms
    hits = 0 # for recall calculation
    query_template = '''
    '''
    single_query = '''{}\n{"size": 50, "query": {"knn": {"sift_vector": {"vector": [%s],"k": 50}}}}\n'''
    for (id,query) in enumerate(queries):
      assert len(query)==dim
      query_str = ""
      for num in query:
        query_str += str(num) + ','
      query_str = query_str[:-1]
      # GET sift-index/_search
      single_query_done = single_query % (query_str)
      query_template += single_query_done
    query_data = query_template
    # print(query_data)
    response = requests.get(url + '_msearch', data=query_data, headers=requestHeaders, auth=HTTPBasicAuth('admin', 'I#vu7bTAHB'), stream=True)
    assert response.status_code==requests.codes.ok
    # print(response.text)
    result = json.loads(response.text)
    # QPS
    total_time = result['took']
    # tooks = []
    # for i in range(len(queries)):
    #   for ele in result['responses']:
    #       tooks.append(int(ele['took']))
    for id in range(len(queries)):
      # Recall
      neighbor_id_from_result = []
      for ele in result['responses'][id]['hits']['hits']:
          neighbor_id_from_result.append(int(ele['_id']))
      assert len(neighbor_id_from_result)==50
      # print("neighbor_id_from_result: ")
      # print(neighbor_id_from_result)
      neighbor_id_gt = neighbors[id][0:50] # topK=50
      # print("neighbor_id_gt")
      # print(neighbor_id_gt)
      hits_q = len(list(set(neighbor_id_from_result) & set(neighbor_id_gt)))
      # print("# hits of this query with topk=50: %d" % hits_q)
      hits += hits_q
    total_time_list.append(total_time)
    hits_list.append(hits)
  print(total_time_list)
  total_time_avg = mean(total_time_list[2:-1])
  hits_avg = mean(hits_list)
  QPS = 1.0 * nq / (total_time_avg / 1000.0)
  recall = 1.0 * hits_avg / (nq * 50)
  print(ef_search, QPS, recall)
