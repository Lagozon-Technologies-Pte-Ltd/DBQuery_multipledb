import pandas as pd
import os
import configure
from operator import itemgetter
from langchain.chains.openai_tools import create_extraction_chain_pydantic 
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI 
#from  langchain_openai.chat_models import with_structured_output

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
llm = ChatOpenAI(model=configure.selected_models, temperature=0)
from typing import List

# @st.cache_data
def get_table_details(selected_subject = 'Demo'):
    # Read the CSV file into a DataFrame
    select_database_table_desc_csv = selected_subject + ".csv"
    path = f'table_files/{select_database_table_desc_csv}'
    # table_description = pd.read_csv("database_table_descriptions.csv")
    table_description = pd.read_csv(path)
    # print("Selected Table description csv is ....." , select_database_table_desc_csv)
    table_docs = []

    # Iterate over the DataFrame rows to create Document objects
    table_details = ""
    for index, row in table_description.iterrows():
        table_details = table_details + "Table Name:" + row['Table'] + "\n" + "Table Description:" + row['Description'] + "\n\n"

    return table_details


class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")

def get_tables(tables: List[Table]) -> List[str]:
    tables  = [table.name for table in tables]
    return tables


# table_names = "\n".join(db.get_usable_table_names())
table_details = get_table_details()
# print("testinf details",table_details, type(table_details))
# table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
#     The permissible tables names are listed below and must be strictly followed:

#     {table_details}

#     Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""
table_details_set_prompt = os.getenv('TABLE_DETAILS_SET_PROMPT')
table_details_prompt=table_details_set_prompt.format(table=table_details)
# print("Table_details_prompt: ", table_details_prompt)
table_chain = {"input": itemgetter("question")} | create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt) | get_tables
# mock_question_test = "How many product view by products in last week"
# table_chain_check = table_chain.invoke({"question":mock_question_test})
# print("test table chain  first mock_question  :" , mock_question_test ,"  Now tables selected:... ",table_chain_check)