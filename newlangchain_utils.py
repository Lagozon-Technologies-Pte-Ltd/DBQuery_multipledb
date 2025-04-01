import os
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
from prompts1 import final_prompt1
from prompts import final_prompt2

from table_details import get_table_details , get_tables , itemgetter , create_extraction_chain_pydantic, Table 
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import configure
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'Cloud_service.json'
db_user = os.getenv("db_user")
db_password = os.getenv("db_password")
db_host=os.getenv("db_host")
#db_warehouse=os.getenv("db_warehouse")
db_database=os.getenv("db_database")
db_port=os.getenv("db_port")
db_schema= os.getenv("db_schema")

client = bigquery.Client()
from sqlalchemy.exc import SQLAlchemyError
def get_postgres_db(selected_subject, chosen_tables):
    print("SELECTED SUB",selected_subject,chosen_tables)
    try:
        print(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}')
        print(db_schema)
        print("Entered try block yoo")
        db = SQLDatabase.from_uri(
            f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}',
            schema=db_schema,
            include_tables=chosen_tables,
            view_support=True,
            sample_rows_in_table_info=1,
            lazy_table_reflection=True
        )
        print("Connection successful")

    except SQLAlchemyError as e:
          print(f"Error connecting to the database: {e}")
    return db
client = bigquery.Client()
def create_bigquery_uri(project_id, dataset_id):
    """Creates a BigQuery URI string."""
    return f"{project_id}.{dataset_id}"


class BigQuerySQLDatabase(SQLDatabase):
    def __init__(self, credentials_path):
        try:
            with open(credentials_path, 'r') as f:
                credentials_info = json.load(f)

            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            self.project_id = credentials_info["project_id"]
            self.client = bigquery.Client(credentials=credentials, project=self.project_id)

        except Exception as e:
            raise ValueError(f"Error loading credentials from JSON file: {e}")

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
# db = BigQuerySQLDatabase("Cloud_service.json")

# table_info = db.get_table_info()
# #Save table_info to a text file
# with open("table_info.txt", "w") as file:
#     file.write(str(table_info))

print("Table info saved successfully to table_info.txt")
# @cache_resource
def get_chain(question, _messages, selected_model, selected_subject, selected_database):
    llm = ChatOpenAI(model=selected_model, temperature=0)

    if selected_database=="GCP":
            db = BigQuerySQLDatabase("Cloud_service.json")
    else:
        table_details = get_table_details(selected_subject)
        print("Selected subject, inside get chain",selected_subject)
        table_details_set_prompt = os.getenv('TABLE_DETAILS_SET_PROMPT')
        table_details_prompt = table_details_set_prompt.format(table=table_details)
        
        table_chain = {"input": itemgetter("question")} | create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt) | get_tables
        print("hiiiii")
        chosen_tables = table_chain.invoke({"question": question})
        print("helllooo")
        db = get_postgres_db(selected_subject, chosen_tables)
        print("foolll")
    print("start",selected_database)
    print("Generate Query Starting")
    if selected_database=="GCP":
        print(selected_database)
        final_prompt=final_prompt1
    else:
        print("kkk")
        final_prompt=final_prompt2    
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
    if selected_database == "GCP":
        chosen_tables = db.get_table_names()
        
    
    return chain, chosen_tables, SQL_Statement, db

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

def invoke_chain(question, messages, selected_model, selected_subject, selected_database):
    print("hiii")
    print(question, messages, selected_model, selected_subject, selected_database)
    try:
        # if not is_relevant(question, table_info):
        #     return "I am DBQuery, a generative AI tool developed at Lagozon Technologies for database query generation. Please ask me queries related to your database.", [], {}, None
        print('Model used:', selected_model)
        history = create_history(messages)
        chain, chosen_tables, SQL_Statement, db = get_chain(question, history.messages, selected_model, selected_subject, selected_database)
        print(f"Generated SQL Statement in newlangchain_utils: {SQL_Statement}")
        SQL_Statement = SQL_Statement.replace("SQL Query:", "").strip()

        response = chain.invoke({"question": question, "top_k": 3, "messages": history.messages})
        print("Question:", question)
        print("Response:", response)
        print("Chosen tables:", chosen_tables)
        alchemyEngine = create_engine(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}')

        tables_data = {}
        for table in chosen_tables:
            query = response["query"]
            print(f"Executing SQL Query: {query}")
            if selected_database=="GCP":
                result_json = db.run(query)
                df = pd.DataFrame(result_json)  # Convert result to DataFrame
                tables_data[table] = df
                break
            else:
                with alchemyEngine.connect() as conn:
                    df = pd.read_sql(
                        sql=query,
                        con=conn.connection
                    )
                # tables_data[table] = pd.DataFrame()
                tables_data[table] = df
                print(table)
                break

        return response, chosen_tables, tables_data, db

    except Exception as e:
        print("Error:", e)
        return "Insufficient information to generate SQL Query.", [], {}, e

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
