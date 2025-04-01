examples = [
    {
        "input": "list all the employees",
        "query": "SELECT * FROM lz_employees",
        "contexts": " | ".join([
            "Table: lz_employees",
            "Columns: employee_id, employee_name, department_id, salary, hire_date, last_name, first_name, email, job_title, full_name",
            "Description: This table has information related to the employees like employee id, hire date, salary, last name, first name, email, department, job title, full name."
        ])
    },
    {
        "input": "list all performance reviews",
        "query": "SELECT * FROM lz_performance_reviews",
        "contexts": " | ".join([
            "Table: lz_performance_reviews",
            "Columns: review_id, employee_id, rating, goals_met, strengths, weaknesses, development_plan",
            "Description: This is a table that stores information about employee performance evaluations, including ratings, goals met, strengths, weaknesses, and development plans."
        ])
    },
    {
        "input": "list all organizations",
        "query": "SELECT * FROM lz_organizations",
        "contexts": " | ".join([
            "Table: lz_organizations",
            "Columns: organization_id, organization_name, description",
            "Description: This is a table that stores information about different functions or departments inside a company, including their names and descriptions."
        ])
    },
    {
        "input": "list all training programs",
        "query": "SELECT * FROM lz_training_programs",
        "contexts": " | ".join([
            "Table: lz_training_programs",
            "Columns: program_id, program_name, description, duration, is_required",
            "Description: This is a table that stores information about training programs offered by an organization, including their names, descriptions, duration, and whether they are required for specific roles."
        ])
    },
    {
        "input": "list all employee training records",
        "query": "SELECT * FROM lz_employee_training",
        "contexts": " | ".join([
            "Table: lz_employee_training",
            "Columns: employee_id, program_id, completion_date, certificate_number",
            "Description: This is a table that stores information about employee participation in training programs, including the employee's ID, the program's ID, the completion date, and the certificate number (if applicable)."
        ])
    },
    {
        "input": "count of training programs by requirement status",
        "query": "SELECT is_required, COUNT(*) as program_count FROM lz_training_programs GROUP BY is_required",
        "contexts": " | ".join([
            "Table: lz_training_programs",
            "Columns: program_id, program_name, is_required",
            "Description: This table stores information about training programs, allowing the count of programs based on their requirement status."
        ])
    },
    {
        "input": "average rating of employee performance reviews",
        "query": "SELECT AVG(rating) as average_rating FROM lz_performance_reviews",
        "contexts": " | ".join([
            "Table: lz_performance_reviews",
            "Columns: review_id, employee_id, rating",
            "Description: This table contains employee performance evaluations, allowing the calculation of the average rating."
        ])
    },
    {
        "input": "list employees who completed training",
        "query": "SELECT employee_id, program_id, completion_date FROM lz_employee_training WHERE completion_date IS NOT NULL",
        "contexts": " | ".join([
            "Table: lz_employee_training",
            "Columns: employee_id, program_id, completion_date",
            "Description: This table tracks employee training completion, including employee ID, program ID, and completion date."
        ])
    },
    {
        "input": "list training programs and their durations",
        "query": "SELECT program_name, duration FROM lz_training_programs",
        "contexts": " | ".join([
            "Table: lz_training_programs",
            "Columns: program_id, program_name, duration",
            "Description: This table contains details of training programs offered by the organization, including program names and durations."
        ])
    },
    {
        "input": "list employees along with their performance reviews",
        "query": "SELECT e.employee_name, p.rating, p.goals_met FROM lz_employees e JOIN lz_performance_reviews p ON e.employee_id = p.employee_id",
        "contexts": " | ".join([
            "Tables: lz_employees, lz_performance_reviews",
            "Columns: employee_id, employee_name, rating, goals_met",
            "Description: This query retrieves employees along with their performance review ratings and goals met."
        ])
    },
    {
        "input": "average salary of employees",
        "query": "SELECT AVG(salary) as average_salary FROM lz_employees",
        "contexts": " | ".join([
            "Table: lz_doctors",
            "Columns: doctor_id, first_name, department_id",
            "Description: This table includes employee details like ID, name, department, and salary, allowing for the calculation of the average salary."
        ])
    },
    {
        "input": "total revenue from sales",
        "query": "SELECT SUM(total_amount) as total_revenue FROM lz_receipts",
        "contexts": " | ".join([
            "Table: lz_receipts",
            "Columns: receipt_id, invoice_id, payment_amount, payment_date",
            "Description: This table records financial transactions, including receipt ID, invoice ID, payment amount, and date."
        ])
    },
    {
        "input": "number of items in stock",
        "query": "SELECT SUM(onhand_quantity) AS total_items_in_stock FROM lz_item_onhand",
        "contexts": " | ".join([
            "Table: lz_item_onhand",
            "Columns: item_id, onhand_quantity, location_id",
            "Description: This table contains inventory data, including item ID, quantity on hand, and location."
        ])
    },
    {
        "input": "number of radiology exams conducted in the last month",
        "query": "SELECT COUNT(*) as exams_last_month FROM lz_radiology_exams WHERE exam_date >= NOW() - INTERVAL '1 month'",
        "contexts": " | ".join([
            "Table: lz_radiology_exams",
            "Columns: exam_id, patient_id, exam_date, exam_type",
            "Description: This table tracks radiology exams, including exam ID, patient ID, exam date, and type of exam."
        ])
    },
    {
        "input": "List all invoices with their corresponding receipts",
        "query": "SELECT i.invoiceid, i.customerid, i.invoicedate, i.duedate, i.totalamount, r.receiptid, r.paymentamount FROM lz_invoices i LEFT JOIN lz_receipts r ON i.invoiceid = r.invoiceid",
        "contexts": " | ".join([
            "Table: lz_invoices, lz_receipts",
            "Columns: invoice_id, customer_id, invoice_date, due_date, total_amount, receipt_id, payment_amount",
            "Description: The `lz_invoices` table contains invoice data, including customer ID, invoice date, due date, and total amount. The `lz_receipts` table contains payment records linked to invoices."
        ])
    },
    {
        "input": "list of doctors by department",
        "query": "SELECT department_id, doctor_name FROM lz_doctors ORDER BY department_id, doctor_name",
        "contexts": " | ".join([
            "Table: lz_employees",
            "Columns: employee_id, first_name, department_id",
            "Description: This table contains information about doctors, including their ID, name, and department."
        ])
    },
    {
        "input": "Get total amount invoiced and total amount paid for each customer",
        "query": "SELECT i.customerid, SUM(i.totalamount) AS total_amount_invoiced, COALESCE(SUM(r.paymentamount), 0) AS total_amount_paid FROM lz_invoices i LEFT JOIN lz_receipts r ON i.invoiceid = r.invoiceid GROUP BY i.customerid",
        "contexts": " | ".join([
            "Table: lz_invoices, lz_receipts",
            "Columns: invoice_id, customer_id, total_amount, payment_amount",
            "Description: The `lz_invoices` table stores invoice details like customer ID, total amount invoiced. The `lz_receipts` table tracks payments made for these invoices."
        ])
    },
    {
    "input": "list all item costs effective after a certain date",
    "query": "SELECT * FROM lz_item_costs WHERE effective_date > '2024-01-01'",
    "contexts": " | ".join([
        "Table: lz_item_costs",
        "Columns: cost_id, item_id, cost_amount, effective_date",
        "Description: This table contains details of item costs, allowing filtering by effective date."
    ])
}, 
    {
    "input": "get average item cost amount",
    "query": "SELECT AVG(cost_amount) AS average_cost FROM lz_item_costs",
    "contexts": " | ".join([
        "Table: lz_item_costs",
        "Columns: average_cost",
        "Description: This table provides cost information, including the average cost amount across all items."
    ])
},
    {
    "input": "list item costs with effective dates",
    "query": "SELECT item_id, cost_amount, effective_date FROM lz_item_costs",
    "contexts": " | ".join([
        "Table: lz_item_costs",
        "Columns: item_id, cost_amount, effective_date",
        "Description: This table tracks the costs of items, along with their effective dates."
    ])
},
    {
    "input": "list item costs by cost type",
    "query": "SELECT cost_type, COUNT(*) AS cost_count FROM lz_item_costs GROUP BY cost_type",
    "contexts": " | ".join([
        "Table: lz_item_costs",
        "Columns: cost_type, cost_count",
        "Description: This table tracks the different types of costs associated with items, including the count of each cost type."
    ])
},
    {
        "input": "List all receipts along with the corresponding invoice details",
        "query": """
            SELECT r.receipt_id, r.payment_amount, i.invoice_id, i.total_amount as invoice_amount
            FROM lz_receipts r
            JOIN lz_invoices i ON r.invoice_id = i.invoice_id
        """,
        "contexts": " | ".join([
            "Table: lz_receipts, lz_invoices",
            "Columns: receipt_id, payment_amount, invoice_id, total_amount",
            "Description: The `lz_receipts` table records payment details linked to invoices stored in the `lz_invoices` table."
        ])
    },
    {
        "input": "List all nurses along with their department names",
        "query": """
            SELECT n.nurse_id, n.nurse_name, d.department_name
            FROM lz_nurses n
            JOIN lz_departments d ON n.department_id = d.department_id
        """,
        "contexts": " | ".join([
            "Table: lz_nurses, lz_departments",
            "Columns: nurse_id, nurse_name, department_id, department_name",
            "Description: The `lz_nurses` table contains information about nurses, while the `lz_departments` table includes department details."
        ])
    },
    {
        "input": "total revenue by customer",
        "query": "SELECT customerid, SUM(totalamount) AS total_revenue FROM lz_invoices GROUP BY customerid",
        "contexts": " | ".join([
            "Table: lz_invoices",
            "Columns: customer_id, total_amount",
            "Description: This table records invoice data, including customer ID and the total amount billed."
        ])
    },
    {
        "input": "list all the invoices in the second financial quarter of 2024",
        "query": "SELECT * FROM lz_invoices WHERE EXTRACT(QUARTER FROM invoicedate) = 2 AND EXTRACT(YEAR FROM invoicedate) = 2024",
        "contexts": " | ".join([
            "Table: lz_invoices",
            "Columns: invoice_id, invoice_date, total_amount",
            "Description: This table stores invoice details, including invoice ID, date, and total amount, allowing filtering by quarter and year."
        ])
    },
    {
        "input": "Find receipts without corresponding invoices",
        "query": "SELECT r.receiptid, r.invoiceid, r.receiptdate, r.paymentamount, r.paymentmethod, r.paymentreference, r.paymentstatus FROM lz_receipts r LEFT JOIN lz_invoices i ON r.invoiceid = i.invoiceid WHERE i.invoiceid IS NULL",
        "contexts": " | ".join([
            "Table: lz_receipts, lz_invoices",
            "Columns: receipt_id, invoice_id, payment_amount, payment_method, payment_status",
            "Description: The `lz_receipts` table stores payment information, while the `lz_invoices` table contains invoice details, enabling the identification of receipts without matching invoices."
        ])
    },
    {
        "input": "List all employees along with their department names",
        "query": """
            SELECT e.employee_id, e.employee_name, d.department_name
            FROM lz_employees e
            JOIN lz_departments d ON e.department_id = d.department_id
        """,
        "contexts": " | ".join([
            "Table: lz_employees, lz_departments",
            "Columns: employee_id, employee_name, department_id, department_name",
            "Description: The `lz_employees` table contains information about employees, while the `lz_departments` table includes department details."
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

