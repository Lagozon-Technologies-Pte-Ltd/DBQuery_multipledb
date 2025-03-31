from fastapi import FastAPI, Form, HTTPException, Query, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import plotly.graph_objects as go
import plotly.express as px
from langchain_openai import ChatOpenAI
import openai, yaml
from configure import gauge_config
import base64
from pydantic import BaseModel
from io import BytesIO, StringIO
import os, csv
import pandas as pd

from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import Field
from langchain_openai import ChatOpenAI
from newlangchain_utils import *
from dotenv import load_dotenv
from state import session_state, session_lock
load_dotenv()  # Load environment variables from .env file
from typing import Optional
from starlette.middleware.sessions import SessionMiddleware  # Correct import
from azure.storage.blob import BlobServiceClient

import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Log the request details
        logging.info(f"Request: {request.method} {request.url}")
        
        # Call the next middleware or endpoint
        response = await call_next(request)
        
        # Log the response details
        logging.info(f"Response status: {response.status_code}")
        
        return response

# Create FastAPI app
app = FastAPI()

# Add the middleware to the app
app.add_middleware(LoggingMiddleware)

app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

# Set up static files and templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
# Azure Blob Storage settings
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME')

# Initialize the BlobServiceClient
try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    print("Blob service client initialized successfully.")
except Exception as e:
    print(f"Error initializing BlobServiceClient: {e}")
    # Handle the error appropriately, possibly exiting the application
    raise  # Re-raise the exception to prevent the app from starting

class ChartRequest(BaseModel):
    """
    Pydantic model for chart generation requests.
    """
    table_name: str
    x_axis: str
    y_axis: str
    chart_type: str

    class Config:  # This ensures compatibility with FastAPI
        json_schema_extra = {
            "example": {
                "table_name": "example_table",
                "x_axis": "column1",
                "y_axis": "column2",
                "chart_type": "Line Chart"
            }
        }

# Initialize OpenAI API key and model
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

subject_areas1 = os.getenv('subject_areas1').split(',')
question_dropdown = os.getenv('Question_dropdown')
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)  # Adjust model as necessary
from table_details import get_table_details  # Importing the function
if 'messages' not in session_state:
    session_state['messages'] = []

class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")

def download_as_excel(data: pd.DataFrame, filename: str = "data.xlsx"):
    """
    Converts a Pandas DataFrame to an Excel file and returns it as a stream.

    Args:
        data (pd.DataFrame): The DataFrame to convert.
        filename (str): The name of the Excel file.  Defaults to "data.xlsx".

    Returns:
        BytesIO:  A BytesIO object containing the Excel file.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)  # Reset the pointer to the beginning of the stream
    return output

def create_gauge_chart_json(title, value, min_val=0, max_val=100, color="blue", subtext="%"):
    """
    Creates a gauge chart using Plotly and returns it as a JSON string.

    Args:
        title (str): The title of the chart.
        value (float): The value to display on the gauge.
        min_val (int): The minimum value of the gauge.  Defaults to 0.
        max_val (int): The maximum value of the gauge.  Defaults to 100.
        color (str): The color of the gauge.  Defaults to "blue".
        subtext (str): The subtext to display below the value. Defaults to "%".

    Returns:
        str: A JSON string representation of the gauge chart.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 18, 'color': 'black'}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color, 'thickness': 1},
            'bgcolor': "white",
            'borderwidth': 0.7,
            'bordercolor': "black",

            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        },
        number={'suffix': subtext, 'font': {'size': 16, 'color': 'gray'}}
    ))

    # Adjust the layout to prevent cropping
    fig.update_layout(
        width=350,  # Increased width
        height=350,  # Increased height
        margin=dict(
            t=50,  # Top margin
            b=50,  # Bottom margin
            l=50,  # Left margin
            r=50   # Right margin
        )

    )
    return fig.to_json()  # Return JSON instead of an image

@app.get("/get-table-columns/")
async def get_table_columns(table_name: str):
    """
    Fetches the columns for a given table from the session state.

    Args:
        table_name (str): The name of the table.

    Returns:
        JSONResponse: A JSON response containing the list of columns or an error message.
    """
    try:
        if "tables_data" not in session_state or table_name not in session_state["tables_data"]:
            raise HTTPException(status_code=404, detail=f"Table {table_name} not found in session.")

        # Retrieve the DataFrame for the specified table
        data_df = session_state["tables_data"][table_name]
        columns = list(data_df.columns)

        return {"columns": columns}
    except Exception as e:
        return JSONResponse(
            content={"error": f"Error fetching columns: {str(e)}"},
            status_code=500
        )

class QueryInput(BaseModel):
    """
    Pydantic model for user query input.
    """
    query: str

@app.post("/add_to_faqs")
async def add_to_faqs(data: QueryInput):
    """
    Adds a user query to the FAQ CSV file.

    Args:
        data (QueryInput): The user query.

    Returns:
        JSONResponse: A JSON response indicating success or failure.
    """
    query = data.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Invalid query!")

    try:
        with open('table_files\Demo_questions.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([query])

        return {"message": "Query added to FAQs successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def generate_chart_figure(data_df: pd.DataFrame, x_axis: str, y_axis: str, chart_type: str):
    """
    Generates a Plotly figure based on the specified chart type.

    Args:
        data_df (pd.DataFrame): The DataFrame containing the data.
        x_axis (str): The column name for the x-axis.
        y_axis (str): The column name for the y-axis.
        chart_type (str): The type of chart to generate.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure, or None if the chart type is unsupported.
    """
    fig = None
    if chart_type == "Line Chart":
        fig = px.line(data_df, x=x_axis, y=y_axis)
    elif chart_type == "Bar Chart":
        fig = px.bar(data_df, x=x_axis, y=y_axis)
    elif chart_type == "Scatter Plot":
        fig = px.scatter(data_df, x=x_axis, y=y_axis)
    elif chart_type == "Pie Chart":
        fig = px.pie(data_df, names=x_axis, values=y_axis)
    elif chart_type == "Histogram":
        fig = px.histogram(data_df, x=x_axis, y=y_axis)
    elif chart_type == "Box Plot":
        fig = px.box(data_df, x=x_axis, y=y_axis)
    elif chart_type == "Heatmap":
        fig = px.density_heatmap(data_df, x=x_axis, y=y_axis)
    elif chart_type == "Violin Plot":
        fig = px.violin(data_df, x=x_axis, y=y_axis)
    elif chart_type == "Area Chart":
        fig = px.area(data_df, x=x_axis, y=y_axis)
    elif chart_type == "Funnel Chart":
        fig = px.funnel(data_df, x=x_axis, y=y_axis)
    return fig

@app.post("/generate-chart/")
async def generate_chart(request: ChartRequest):
    """
    Generates a chart based on the provided request data.
    """
    try:
        table_name = request.table_name
        x_axis = request.x_axis
        y_axis = request.y_axis
        chart_type = request.chart_type

        print(f"Received Request: {request.dict()}")

        if "tables_data" not in session_state or table_name not in session_state["tables_data"]:
            return JSONResponse(
                content={"error": f"No data found for table {table_name}"},
                status_code=404
            )

        data_df = session_state["tables_data"][table_name]
        print(f"Table {table_name} data: {data_df.head()}")  # Print first few rows of the DataFrame
        print(f"X-axis data type: {data_df[x_axis].dtype}")
        print(f"Y-axis data type: {data_df[y_axis].dtype}")

        # Explicit type conversion (example)
        try:
            data_df[y_axis] = pd.to_numeric(data_df[y_axis], errors='coerce')
            data_df = data_df.dropna(subset=[y_axis])
        except Exception as e:
            print(f"Error converting data to numeric: {e}")
            return JSONResponse(
                content={"error": f"Error converting data to numeric: {str(e)}"},
                status_code=400
            )

        print(f"Generating {chart_type} for Table: {table_name}, X: {x_axis}, Y: {y_axis}")

        fig = generate_chart_figure(data_df, x_axis, y_axis, chart_type)

        if fig:
            chart_json = fig.to_json()
            #print(chart_json) # consider limiting this output as it can be very large
            return JSONResponse(content={"chart": chart_json})
        else:
            return JSONResponse(content={"error": "Unsupported chart type selected."}, status_code=400)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(f"Chart generation error: {e}")
        return JSONResponse(
            content={"error": f"An error occurred while generating the chart: {str(e)}"},
            status_code=500
        )

@app.get("/download-table/")
async def download_table(table_name: str):
    """
    Downloads a table as an Excel file.

    Args:
        table_name (str): The name of the table to download.

    Returns:
        StreamingResponse: A streaming response containing the Excel file.
    """
    # Check if the requested table exists in session state
    if "tables_data" not in session_state or table_name not in session_state["tables_data"]:
        raise HTTPException(status_code=404, detail=f"Table {table_name} data not found.")

    # Get the table data from session_state
    data = session_state["tables_data"][table_name]

    # Generate Excel file
    output = download_as_excel(data, filename=f"{table_name}.xlsx")

    # Return the Excel file as a streaming response
    response = StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response.headers["Content-Disposition"] = f"attachment; filename={table_name}.xlsx"
    return response

@app.post("/transcribe-audio/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribes an audio file using OpenAI's Whisper API.

    Args:
        file (UploadFile): The audio file to transcribe.

    Returns:
        JSONResponse: A JSON response containing the transcription or an error message.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Missing OpenAI API Key.")
        audio_bytes = await file.read()
        audio_bio = BytesIO(audio_bytes)
        audio_bio.name = "audio.webm"

        # Fix: Using OpenAI API correctly
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bio
        )

        # Fix: Access `transcript.text` instead of treating it as a dictionary
        return {"transcription": transcript.text}

    except Exception as e:
        return JSONResponse(content={"error": f"Error transcribing audio: {str(e)}"}, status_code=500)

@app.get("/get_questions")
async def get_questions(subject: str):
    """
    Fetches questions from a CSV file in Azure Blob Storage based on the selected subject.

    Args:
        subject (str): The subject to fetch questions for.

    Returns:
        JSONResponse: A JSON response containing the list of questions or an error message.
    """
    csv_file_name = f"table_files/{subject}_questions.csv"
    blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=csv_file_name)

    try:
        # Check if the blob exists
        if not blob_client.exists():
            print(f"file not found {csv_file_name}")
            return JSONResponse(
                content={"error": f"The file {csv_file_name} does not exist."}, status_code=404
            )

        # Download the blob content
        blob_content = blob_client.download_blob().content_as_text()

        # Read the CSV content
        questions_df = pd.read_csv(StringIO(blob_content))
        
        if "question" in questions_df.columns:
            questions = questions_df["question"].tolist()
        else:
            questions = questions_df.iloc[:, 0].tolist()

        return {"questions": questions}

    except Exception as e:
        return JSONResponse(
            content={"error": f"An error occurred while reading the file: {str(e)}"}, status_code=500
        )# Function to load prompts from YAML

def load_prompts():
    """
    Loads prompts from the chatbot_prompt.yaml file.

    Returns:
        dict: A dictionary containing the loaded prompts.
    """
    try:
        with open("chatbot_prompt.yaml", "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error reading prompts file: {e}")
        return {}

# Load prompts at startup
PROMPTS = load_prompts()

@app.get("/get-tables/")
async def get_tables(selected_section: str):
    """
    Fetches table names for a given section using the get_table_details function.

    Args:
        selected_section (str): The section to fetch tables for.

    Returns:
        JSONResponse: A JSON response containing the list of table names.
    """
    # Fetch table details for the selected section
    table_details = get_table_details(selected_section)
    # Extract table names dynamically
    tables = [line.split("Table Name:")[1].strip() for line in table_details.split("\n") if "Table Name:" in line]
    # Return tables as JSON
    return {"tables": tables}

if 'messages' not in session_state:
    session_state['messages'] = []

@app.post("/submit")
async def submit_query(
    request: Request,
    section: str = Form(...),
    user_query: str = Form(...),
    page: int = Query(1),
    records_per_page: int = Query(10),
    model: Optional[str] = Form("gpt-4o-mini")
):
    if user_query.lower() == 'break':
# Capture current state before reset
        response_data = {
            "user_query": user_query,
            "chat_response": "Session restarted",
            "history": session_state['messages'] + [{"role": "assistant", "content": "Session restarted"}]
        }
        
        # Clear session state
        session_state.clear()
        session_state['messages'] = []  # Reinitialize messages array
        
        return JSONResponse(content=response_data)        
    selected_subject = section
    session_state['user_query'] = user_query

    # Append user's message to chat history
    session_state['messages'].append({"role": "user", "content": user_query})

    chat_history = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in session_state['messages'][-10:]
    )  # Keep last 10 messages for better context
    print("chat history: ", chat_history)
    try:
        # **Step 1: Invoke Unified Prompt**
        unified_prompt = PROMPTS["unified_prompt"].format(user_query=user_query, chat_history=chat_history)
        response = llm.invoke(unified_prompt).content.strip()
        print("response: ",response)
        # **Step 2: Handle Response**
        if response.lower() != "database":
            # ✅ Answer found in history → Return it directly
            session_state['messages'].append({"role": "assistant", "content": response})
            return JSONResponse(content={
                "user_query": user_query,
                "chat_response": response,
                "history": session_state['messages']
            })

        # **Step 3: Continue to Database Query if Needed**
        response, chosen_tables, tables_data, agent_executor = invoke_chain(
            user_query, session_state['messages'], model, selected_subject
        )

        if isinstance(response, str):
            session_state['generated_query'] = response
        else:
            session_state['chosen_tables'] = chosen_tables
            session_state['tables_data'] = tables_data
            sql_query = response.get("query", "")
            session_state['generated_query'] = sql_query

        # **Step 4: Generate Insights (if data exists)**
        chat_insight = None
        if chosen_tables:
            data_preview = tables_data[chosen_tables[0]].head(5).to_string(index=False) if chosen_tables else "No Data"
            
            insights_prompt = PROMPTS["insights_prompt"].format(
                sql_query=sql_query,
                table_data=tables_data
            )

            chat_insight = llm.invoke(insights_prompt).content

        # Append AI's response to chat history
        session_state['messages'].append({
            "role": "assistant",
            "content": f" {chat_insight}\n\n"
        })
        for table_name, df in tables_data.items():
            for col in df.select_dtypes(include=['number']).columns:
                tables_data[table_name][col] = df[col].apply(format_number)        # **Step 5: Prepare Table Data**
        tables_html = prepare_table_html(tables_data, page, records_per_page)

        # **Step 6: Append Table Data to Chat History**
        if tables_html:
            session_state['messages'].append({
                "role": "table_data",
                "content": f"\n{tables_data}"
            })

        # **Step 7: Return Response**
        response_data = {
            "user_query": session_state['user_query'],
            "query": session_state['generated_query'],
            "tables": tables_html,
            "chat_response": chat_insight,
            "history": session_state['messages']
        }
        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the prompt: {str(e)}")
# Replace APIRouter with direct app.post
def format_number(x):
    if x.is_integer():
        return f"{int(x):d}"
    else:
        return f"{x:.1f}"
@app.post("/reset-session")
async def reset_session():
    """
    Resets the session state by clearing the session_state dictionary.
    """
    global session_state
    with session_lock:
        session_state.clear()
        session_state['messages'] = []
    return {"message": "Session state cleared successfully"}, 200

def prepare_table_html(tables_data, page, records_per_page):
    """
    Prepares HTML for displaying table data with pagination.

    Args:
        tables_data (dict): A dictionary of table names and their corresponding DataFrames.
        page (int): The current page number.
        records_per_page (int): The number of records to display per page.

    Returns:
        list: A list of dictionaries containing table name, HTML, and pagination information.
    """
    tables_html = []
    for table_name, data in tables_data.items():
        total_records = len(data)
        total_pages = (total_records + records_per_page - 1) // records_per_page
        html_table = display_table_with_styles(data, table_name, page, records_per_page)
        print("html table: ", data)
        print("end of table data")
        tables_html.append({
            "table_name": table_name,
            "table_html": html_table,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "records_per_page": records_per_page,
            }
        })
    return tables_html

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, subject: Optional[str] = None  # Capture the selected subject
):
    """
    Renders the root HTML page.

    Args:
        request (Request): The incoming request.

    Returns:
        TemplateResponse: The rendered HTML template.
    """
    # Extract table names dynamically
    tables = []
     # Fetch questions for the selected subject
    if subject:
        questions_response = await get_questions(subject)  # Use your existing function
        if "questions" in questions_response:
            questions = questions_response["questions"]
        else:
            questions = []
    else:
        questions = [] # Default: No subject selected

    # Pass dynamically populated dropdown options to the template
    return templates.TemplateResponse("index.html", {
        "request": request,
        "section": subject_areas1,
        "tables": tables,   
        "questions": questions              # Table dropdown based on database selection
    })

# Table data display endpoint
def display_table_with_styles(data, table_name, page_number, records_per_page):
    """
    Displays a Pandas DataFrame as an HTML table with custom styles and pagination.

    Args:
        data (pd.DataFrame): The DataFrame to display.
        table_name (str): The name of the table.
        page_number (int): The current page number.
        records_per_page (int): The number of records to display per page.

    Returns:
        str: An HTML string representation of the styled table.
    """
    start_index = (page_number - 1) * records_per_page
    end_index = start_index + records_per_page
    page_data = data.iloc[start_index:end_index]
    # Reset index and add 1 to start from 1 instead of 0
    page_data = page_data.reset_index(drop=True)
    page_data.index = page_data.index + 1
    styled_table = page_data.style.set_table_attributes('style="border: 2px solid black; border-collapse: collapse;"') \
        .set_table_styles(
            [{
                'selector': 'th',
                'props': [('background-color', '#333'),
                          ('color', 'white')]
            },
                {
                    'selector': 'td',
                    'props': [('border', '1px solid black')]
                }
            ])
    return styled_table.to_html()

@app.get("/get_table_data")
async def get_table_data(
    table_name: str = Query(...),
    page_number: int = Query(1),
    records_per_page: int = Query(10),
):
    """Fetch paginated and styled table data."""
    try:
        # Check if the requested table exists in session state
        if "tables_data" not in session_state or table_name not in session_state["tables_data"]:
            raise HTTPException(status_code=404, detail=f"Table {table_name} data not found.")

        # Retrieve the data for the specified table
        data = session_state["tables_data"][table_name]
        total_records = len(data)
        total_pages = (total_records + records_per_page - 1) // records_per_page

        # Ensure valid page number
        if page_number < 1 or page_number > total_pages:
            raise HTTPException(status_code=400, detail="Invalid page number.")

        # Slice data for the requested page
        start_index = (page_number - 1) * records_per_page
        end_index = start_index + records_per_page
        page_data = data.iloc[start_index:end_index]

        # Style the table as HTML
        styled_table = (
            page_data.style.set_table_attributes('style="border: 2px solid black; border-collapse: collapse;"')
            .set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#333'), ('color', 'white'), ('font-weight', 'bold'), ('font-size', '16px')]},
                {'selector': 'td', 'props': [('border', '2px solid black'), ('padding', '5px')]},
            ])
            .to_html(escape=False)  # Render as HTML
        )

        return {
            "table_html": styled_table,
            "page_number": page_number,
            "total_pages": total_pages,
            "total_records": total_records,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating table data: {str(e)}")
