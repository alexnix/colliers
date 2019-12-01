from pymongo import MongoClient
from pprint import pprint

client = MongoClient("mongodb://192.168.1.41:27017/")
db = client.up

print(db.up.count_documents({}))
