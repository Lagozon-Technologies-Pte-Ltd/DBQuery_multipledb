import os, base64
import pandas as pd
from google.cloud import bigquery

from dotenv import load_dotenv
load_dotenv()

#table_details_prompt = os.getenv('TABLE_DETAILS_PROMPT')
# Change if your schema is different
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
# LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
# LANGCHAIN_ENDPOINT=os.getenv("LANGCHAIN_ENDPOINT")


from langchain_community.utilities.sql_database import SQLDatabase
#from langchain.agents import create_sql_agent
#from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.memory import ChatMessageHistory
from operator import itemgetter
from google.oauth2 import service_account
import json
from urllib.parse import quote_plus


from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from table_details import table_chain as select_table
from prompts import final_prompt, answer_prompt, few_shot_prompt
from table_details import get_table_details , get_tables , itemgetter , create_extraction_chain_pydantic, Table 
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import configure
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'Cloud_service.json'
# client = bigquery.Client()
def create_bigquery_uri(project_id, dataset_id):
    """Creates a BigQuery URI string."""
    return f"{project_id}.{dataset_id}"


class BigQuerySQLDatabase(SQLDatabase):
    def __init__(self):
        try:
            # Create credentials dictionary from environment variables
            credentials_info = {
                "type": os.getenv('GOOGLE_CREDENTIALS_TYPE'),
                "project_id": os.getenv('GOOGLE_CREDENTIALS_PROJECT_ID'),
                "private_key_id": os.getenv('GOOGLE_CREDENTIALS_PRIVATE_KEY_ID'),
                "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDln+0curmestQu\nEjLJWLY5YkCdmhlEZfWvCapN41hO6mS6nwVeYQw4ICP8ltbdsAZrsmzVgtf3GC+G\ngL99wG5WeEd/F1XPTemg8mKbMAf67nGWMc3z5yV3U4sGEnDglCVh1gHhDQC/px2K\nWopLVC/F46zQ+ERj8RjFCXExuZrzCExuFvRrT6dalDOqH8XFLeonnLoqJkPVgjvW\nfuuihW5pMiOfGyXksabfOc1GAzt4Ixbp0rsUL10ZqTPz+FOQ4WeJcs1slgRSQxHC\nmnTKx5kAT8MHEChGzhX9/BHDDzjTZL5isEybWjbKuUEcqCpc1FajFMT8NSDayifz\nEtHnxHhjAgMBAAECggEAAJYeec2r1d5/1Ttx3F3qf59TUJ/9qjwZu0SQQf2DOSvy\nuLHbYYGcUupehJ3LIBmiTIxyvEKrwibe3eJdLo5jQqZccY3OZbnN93T+8lHAMs4F\njkpRKj/WB0dF1uImLDXaTAPfM1lezVsgO71ESZ6L53fKBYrSXGLP/bVOfbcJTuwn\nFjAgNdpj2xYl+G9B+qkuNBHZ7uVnQ3w2l4zvRAWIIwtRj1qjCe7ynac9xizkrEMI\nae1WCKCZQbJGOvOl4Mu00cVvfspwzHfQZwkn+dVjN2+HNQTbzsM14CzDTmXGjD6e\n5/s3OYj7Tt4lV/PIVsf/y/zz3mtVV5D73yWQiZbiNQKBgQD9HfRuKnrKFrnyTy31\nRkSqZTfZh2smRqiR4RZssgZUCKD5GZtQ3/opWkh2HSBdQY8tLkxiu7wJ9WKmHMVV\nnUANqcBxXwsaLdMVEt4C7Y3aav8owIn+rLxD0BuQkjbX+7cx0UTnNhmg97HpYJr5\nNV+xF2LyviTemPpviWI2W6N9FwKBgQDoPXjR+L8ow0Sxhu5IjLWWp86X4KXQOCuY\n/Qbk+L3ibM8DRpgZ+nwH9zDWcGIS3Kk5t8pIQSYbthYBugkekUvtCt2dRyxIPLK9\nXnaCJFSbtpd1aaII/YF6Gp0yaap0B3+x9L4w1UrvLHK3xUcVdeb3DDCj0IVAqBg9\nqtLoktbmlQKBgQC1cTqdmh/pK79hnjbAov1n9CTD71n01yPRZrvPcRIuPP0/c4at\nw9CswgY9fQWNNAixh4XEJPVXYiq0Dt26UH3xDWVhH5Ny0bSFX7/781QDZT3Bdbu1\n7xcJuW15BgcAbnVU5cFxyIs4ozZKqDCPQh51cOFCRuFhG+IyABaCBtC8QwKBgHvw\nam0sIeBALYXMa5geN76Z+WAGTJdNkr7Hsgk6UiPnS6cE4qFikxSxL8gRG9XTGyCp\nW/OpiQva5e2v+bPteKadWN3ZoOFAO2diZT5Y4ypijHvljsrbd2DRmTjROV1IrzYq\nVeG7wozXnLVEPAZQ8JzBTafu3V4/Fwi6BGqICtXtAoGAb1QEQxRfq87q2q7DxIbm\nlxooi07TB1eevVw6r2qNRQQ5DHF+vb65Tw9ZV3E0g8/fJRD2gFC+yxgfI3iUVyyh\nIBBjKgCJOgp6zOS1L+RTNQswXxxLw+5B9j/oArHZ24j7YtKPLr+bcTNypzXn8dh8\n1U/HqFUTo1bsy8Pu35MXyco=\n-----END PRIVATE KEY-----",
                "client_email": os.getenv('GOOGLE_CREDENTIALS_CLIENT_EMAIL'),
                "client_id": os.getenv('GOOGLE_CREDENTIALS_CLIENT_ID'),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/bqserviceacc%40prateekproject-450509.iam.gserviceaccount.com",
                "universe_domain": "googleapis.com"
            }

            # Load credentials from dictionary
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=["https://www.googleapis.com/auth/bigquery"]
            )

            self.project_id = credentials_info["project_id"]
            self.client = bigquery.Client(credentials=credentials, project=self.project_id)

        except Exception as e:
            raise ValueError(f"Error loading credentials: {e}")
    def run(self, command: str):
        """Executes a SQL query and returns results as JSON."""
        try:
            query_job = self.client.query(command)
            results = query_job.result()
            return [dict(row.items()) for row in results]
        except Exception as e:
            return f"Error executing SQL command: {e}"

    def get_table_names(self):
        """Returns all available tables in the project."""
        tables_list = []
        datasets = list(self.client.list_datasets())
        for dataset in datasets:
            dataset_id = dataset.dataset_id
            tables = self.client.list_tables(dataset_id)
            for table in tables:
                tables_list.append(f"{dataset_id}.{table.table_id}")
        return tables_list

    def get_table_info(self, table_names=None):
        """Returns schema information for given tables."""
        if table_names is None:
            table_names = self.get_table_names()

        schema_info = ""
        for table_name in table_names:
            try:
                dataset_id, table_id = table_name.split(".")
                table_ref = self.client.dataset(dataset_id).table(table_id)
                table = self.client.get_table(table_ref)

                schema_info += f"\nTable: {table_name}\nColumns:\n"
                for column in table.schema:
                    schema_info += f"  {column.name} ({column.field_type}) {'NULLABLE' if column.is_nullable else 'NOT NULLABLE'}\n"
            except Exception as e:
                schema_info += f"Error getting schema for table {table_name}: {e}\n"

        return schema_info
db = BigQuerySQLDatabase()

table_info = db.get_table_info()
#Save table_info to a text file
with open("table_info.txt", "w") as file:
    file.write(str(table_info))

print("Table info saved successfully to table_info.txt")
# @cache_resource
def get_chain(question, _messages, selected_model, selected_subject='Demo'):
    llm = ChatOpenAI(model=selected_model, temperature=0)

    db = BigQuerySQLDatabase()  # Use the correct class

    print("Generate Query Starting")
    generate_query = create_sql_query_chain(llm, db, final_prompt)
    SQL_Statement = generate_query.invoke({"question": question, "messages": _messages})
    print(f"Generated SQL Statement before execution: {SQL_Statement}")

    # Override QuerySQLDataBaseTool validation
    class CustomQuerySQLDatabaseTool(QuerySQLDataBaseTool):
        def __init__(self, db):
            if not isinstance(db, SQLDatabase):
                raise ValueError("db must be an instance of SQLDatabase")
            super().__init__(db=db)

    execute_query = CustomQuerySQLDatabaseTool(db=db)
    
    chain = (
        RunnablePassthrough.assign(table_names_to_use=lambda _: db.get_table_names()) |  # Get table names
        RunnablePassthrough.assign(query=generate_query).assign(
            result=itemgetter("query")
        )
    )

    return chain, db.get_table_names(), SQL_Statement, db

# def read_table_info(file_path):
#     """
#     Reads the table schema from a text file as a single string.
#     """
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             return file.read()  # Read full content as a string
#     except Exception as e:
#         print(f"Error reading table info: {e}")
#         return ""
# table_info_text = read_table_info("table_info.txt")  # Load the entire text of the file

def invoke_chain(question, messages, selected_model, selected_subject):
    try:
        # if not is_relevant(question, table_info):
        #     return "I am DBQuery, a generative AI tool developed at Lagozon Technologies for database query generation. Please ask me queries related to your database.", [], {}, None
        print('Model used:', selected_model)
        history = create_history(messages)
        chain, chosen_tables, SQL_Statement, db = get_chain(question, history.messages, selected_model, selected_subject)
        print(f"Generated SQL Statement: {SQL_Statement}")
        SQL_Statement = SQL_Statement.replace("SQL Query:", "").strip()

        response = chain.invoke({"question": question, "top_k": 3, "messages": history.messages})
        print("Question:", question)
        print("Response:", response)
        print("Chosen tables:", chosen_tables)

        tables_data = {}
        for table in chosen_tables:
            query = response["query"]
            print(f"Executing SQL Query: {query}")

            result_json = db.run(query)
            df = pd.DataFrame(result_json)  # Convert result to DataFrame
            tables_data[table] = df
            break

        return response, chosen_tables, tables_data, db

    except Exception as e:
        print("Error:", e)
        return "Insufficient information to generate SQL Query.", [], {}, None

def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history

def escape_single_quotes(input_string):
    return input_string.replace("'", "''")
