examples = [
 {
    "input": "show total retail volume in financial year 2024",
    "query": """
        SELECT 
            SUM(b.`Retail Volume`) AS `Total Retail Volume`
        FROM DS_sales_data.billing_data b
        WHERE b.`Date` BETWEEN DATE('2024-04-01') AND DATE('2025-03-31');   
        """,
    "contexts": " | ".join([
        "Table: DS_sales_data.billing_data",
        "Columns: Date, Retail Volume",
        "Description: This table contains vehicle retail volume data categorized by date."
    ])
},
 {
    "input": "get monthly test drives for 2024",
    "query": """
        SELECT 
            FORMAT_DATE('%B %Y', b.`Date`) AS `Month`, 
            SUM(b.`Test Drive`) AS `Total Test Drives`
        FROM DS_sales_data.billing_data b
        WHERE b.`Date` BETWEEN DATE('2024-04-01') AND DATE('2025-03-31')
        GROUP BY `Month`
        ORDER BY MIN(b.`Date`);
    """,
    "contexts": " | ".join([
        "Table: DS_sales_data.billing_data",
        "Columns: Date, Test Drive",
        "Description: This table records test drive data, including the number of test drives conducted."
    ])
},
 {
    "input": "show total bookings and billings for model XUV700",
    "query": """
        SELECT 
            p.`Model Name`, 
            SUM(b.`Open Booking`) AS `Total Bookings`, 
            SUM(b.`Billing Volume`) AS `Total Billings`
        FROM DS_sales_data.billing_data b
        JOIN DS_sales_data.product_hierarchy p ON b.`Model ID` = p.`Model ID`
        WHERE LOWER(p.`Model Name`) = LOWER('XUV700') 
        AND b.`Date` BETWEEN DATE('2024-04-01') AND DATE('2025-03-31')
        GROUP BY p.`Model Name`;
        """,
    "contexts": " | ".join([
        "Table: DS_sales_data.billing_data, DS_sales_data.product_hierarchy",
        "Columns: Model ID, Open Booking, Billing Volume, Model Name",
        "Description: This query retrieves the total bookings and billings for a given vehicle model."
    ])
},
  {
        "input": "What is the total retail volume for North in Jul 2024",
        "query": """
            SELECT 
                SUM(b.`Retail Volume`) AS `Total Retail Volume`
            FROM DS_sales_data.billing_data b
            JOIN DS_sales_data.sales_person_hierarchy s ON b.`RSM ID` = s.`RSM ID`
            WHERE s.`Zone Name` = 'North'
            AND b.`Date` BETWEEN DATE('2024-07-01') AND DATE('2024-07-31');
        """,
        "contexts": " | ".join([
            "Table: DS_sales_data.billing_data, DS_sales_data.sales_person_hierarchy",
            "Columns: RSM ID, Retail Volume, Zone Name, Date",
            "Description: This query calculates the total retail volume for a specific zone and month."
        ])
    },
    {
        "input": "Get the monthly billing volume for each model in 2024",
        "query": """
            SELECT 
                p.`Model Name`, 
                FORMAT_DATE('%B %Y', b.`Date`) AS `Month`, 
                SUM(b.`Billing Volume`) AS `Total Billing Volume`
            FROM DS_sales_data.billing_data b
            JOIN DS_sales_data.product_hierarchy p ON b.`Model ID` = p.`Model ID`
            WHERE b.`Date` BETWEEN DATE('2024-04-01') AND DATE('2025-03-31')
            GROUP BY p.`Model Name`, `Month`
            ORDER BY p.`Model Name`, MIN(b.`Date`);
        """,
        "contexts": " | ".join([
            "Table: DS_sales_data.billing_data, DS_sales_data.product_hierarchy",
            "Columns: Model ID, Billing Volume, Model Name, Date",
            "Description: This query provides the monthly billing volume for each vehicle model."
        ])
    },
    {
        "input": "Show total test drives for each zone in financial year 2024",
        "query": """
            SELECT 
                s.`Zone Name`, 
                SUM(b.`Test Drive`) AS `Total Test Drives`
            FROM DS_sales_data.billing_data b
            JOIN DS_sales_data.sales_person_hierarchy s ON b.`RSM ID` = s.`RSM ID`
            WHERE b.`Date` BETWEEN DATE('2024-04-01') AND DATE('2025-03-31')
            GROUP BY s.`Zone Name`
            ORDER BY `Total Test Drives` DESC;
        """,
        "contexts": " | ".join([
            "Table: DS_sales_data.billing_data, DS_sales_data.sales_person_hierarchy",
            "Columns: RSM ID, Test Drive, Zone Name, Date",
            "Description: This query calculates the total test drives for each zone during a financial year."
        ])
    },
    {
        "input": "Compare monthly retail volume growth for North zone between 2023 and 2024",
        "query": """
            WITH MonthlySales AS (
                SELECT 
                    s.`Zone Name`, 
                    FORMAT_DATE('%B %Y', b.`Date`) AS `Month`, 
                    DATE_TRUNC(b.`Date`, MONTH) AS `Month_Start`,
                    SUM(b.`Retail Volume`) AS `Retail Volume`
                FROM DS_sales_data.billing_data b
                JOIN DS_sales_data.sales_person_hierarchy s ON b.`RSM ID` = s.`RSM ID`
                WHERE s.`Zone Name` = 'North'
                AND b.`Date` BETWEEN DATE('2023-04-01') AND DATE('2025-03-31')
                GROUP BY s.`Zone Name`, `Month`, `Month_Start`
            )
            SELECT 
                `Month`, 
                `Retail Volume` AS `Current_Retail_Volume`,
                LAG(`Retail Volume`) OVER (ORDER BY `Month_Start`) AS `Previous_Retail_Volume`,
                ( (`Retail Volume` - LAG(`Retail Volume`) OVER (ORDER BY `Month_Start`)) 
                  / LAG(`Retail Volume`) OVER (ORDER BY `Month_Start`) ) * 100 AS `Growth_Percentage`
            FROM MonthlySales
            ORDER BY `Month_Start`;
        """,
        "contexts": " | ".join([
            "Table: DS_sales_data.billing_data, DS_sales_data.sales_person_hierarchy",
            "Columns: RSM ID, Retail Volume, Zone Name, Date",
            "Description: This query compares the monthly retail volume growth for a specific zone between two financial years."
        ])
    }



]



from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings


def get_example_selector():
    """
    Returns a SemanticSimilarityExampleSelector object initialized with the given examples.
    """
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,  # Ensure `examples` is a list of dictionaries
        OpenAIEmbeddings(),
        Chroma,
        k=1,
        input_keys=["input"],  # Ensure that 'input' is a key in the examples
    )

    return example_selector

