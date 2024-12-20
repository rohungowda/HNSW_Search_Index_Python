import psycopg2
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

mongo_url = os.getenv('MONGODB_URL')
postgress_user, postgress_password, postgress_host, postgress_port = os.getenv('USER_NAME'), os.getenv('PASSWORD'), os.getenv('HOST'), os.getenv('PORT')
database_name = "news"

def mongo_connector():
    try:
        client = MongoClient(mongo_url)
        db = client[database_name]
        return db
    except Exception as error:
        print("Error while connecting to MongoDB", error)
        exit(1)



def psql_connector():
    try:

        conn = psycopg2.connect(
            dbname= database_name,
            user= postgress_user,
            password= postgress_password,
            host= postgress_host,
            port= postgress_port
        )

        return conn
    except Exception as error:
        print("Error while connecting to PostgreSQL", error)
        exit(1)