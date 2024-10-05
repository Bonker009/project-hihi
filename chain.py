import os
import matplotlib.pyplot as plt
import pandas as pd
from groq import Groq
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

# Fetch Groq API key from environment
groq_api_key = os.environ.get("GROQ_API_KEY")

# Check for missing API key
if not groq_api_key:
    raise EnvironmentError("GROQ_API_KEY environment variable not set.")

# Initialize Groq client with API key
client = Groq(api_key=groq_api_key)

# Database connection (PostgreSQL)
db = SQLDatabase.from_uri("postgresql://postgres:1234@localhost:5432/oauth2")

# Initialize LLM models
llm_groq = ChatGroq(model="llama3-8b-8192")

# Print available tables for verification
print(f"Database dialect: {db.dialect}")
print(f"Usable tables: {db.get_usable_table_names()}")

# SQL query generation chain using LangChain
chain = create_sql_query_chain(llm_groq, db)

# Answer prompt template
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
)

# Create a prompt template for detecting visualization requests
visualization_detection_prompt_template = PromptTemplate.from_template(
    """
    Given the following user input, determine if the user is asking for a data visualization (such as a chart, graph, or plot).
    Please respond with either 'yes' if visualization is requested, or 'no' if no visualization is requested.

    User input: "{input}"
    Visualization requested: """
)

# PromptTemplate for warning about sensitive information or modification requests
sensitive_info_prompt_template = PromptTemplate.from_template(
    """
    The user query might involve sensitive information or modifying the database. The query is: "{question}"

    Please check if the query involves:
    1. Sensitive information such as passwords, authentication tokens, or security details.
    2. SQL commands that can modify the database or tables, such as INSERT, UPDATE, DELETE, DROP.

    If any of the above applies, respond with a warning message and suggest the user refine their query.
    Otherwise, simply return 'safe to proceed'.
    """
)


def refine_prompt(llm, question):
    """Check if the prompt needs refinement (security, sensitive data, or modifying commands)."""
    # List of keywords to detect sensitive information or modification requests
    sensitive_keywords = [
        "security",
        "password",
        "authentication",
        "token",
        "credentials",
    ]
    modification_keywords = ["insert", "update", "delete", "drop", "alter", "truncate"]

    # Perform a simple keyword check first
    for keyword in sensitive_keywords + modification_keywords:
        if keyword.lower() in question.lower():
            print(
                f"Potential sensitive or modifying operation detected in the question. Refining the query..."
            )

            # Use LLM to further analyze the prompt via a prompt template
            prompt = sensitive_info_prompt_template.format(question=question)
            human_message = HumanMessage(content=prompt)
            response = llm.invoke([human_message])

            # Check the LLM's response and decide if the query should be refined
            if "safe to proceed" in response.content.lower():
                print("LLM confirmed the query is safe to proceed.")
                return True
            else:
                print(f"LLM response: {response.content}")
                return False

    # If no sensitive or modifying keywords are detected, proceed normally
    return True


def check_for_visualization_request(llm, user_input):
    """Use the LLM to check if the user's input is asking for a chart or visualization."""
    # Format the prompt with the user's input
    prompt = visualization_detection_prompt_template.format(input=user_input)

    # Send the prompt to the LLM
    human_message = HumanMessage(content=prompt)
    response = llm.invoke([human_message])

    # Check the response content to determine if visualization is requested
    is_visualization_requested = response.content.strip().lower()

    # Return True if the LLM detects a visualization request, otherwise return False
    if is_visualization_requested == "yes":
        return True
    return False


def ask_llm_to_generate_code(llm, sql_query, result):
    """Ask the LLM to generate Python code for visualization, without extra explanations or markdown formatting."""
    code_prompt = f"""
    Based on the following SQL query result, generate only the Python code necessary to visualize the data.
    Do not include any extra explanations, markdown backticks, or comments.
    
    SQL Query: {sql_query}
    SQL Result: {result}

    The Python code should use libraries like matplotlib and pandas to create a suitable chart (bar, line, scatter, etc.).
    The chart should be saved as a PNG file in the current working directory (os.getcwd()) and use uuid for the file name, save the file in a folder named 'seyha' if it does not exist, create one.
    Ensure the chart has appropriate labels and a title.
    """
    # Send the prompt to the LLM and get the response
    human_message = HumanMessage(content=code_prompt)
    response = llm.invoke([human_message])

    # Clean the generated code by removing backticks, markdown, and non-code text
    cleaned_code = response.content

    # Strip any unwanted introduction text before the actual code
    start_code = cleaned_code.find("import")  # Find the start of the actual code
    cleaned_code = cleaned_code[start_code:]  # Remove everything before "import"

    # Remove backticks or any extra markdown text
    cleaned_code = cleaned_code.replace("```python", "").replace("```", "").strip()

    print("cleaned python code:\n", cleaned_code, " end")

    return cleaned_code


def execute_generated_code(generated_code):
    """Execute the generated Python code."""
    try:
        # Dynamically execute the generated code
        exec(generated_code, globals())
        print("Python code executed successfully.")
    except Exception as e:
        print(f"Error executing generated code: {e}")
        print(f"Generated code:\n{generated_code}")


# Main loop to take input from the console
while True:
    try:
        # Take question input from the user
        question = input("Enter your question (or type 'exit' to quit): ")

        # Exit loop if user types 'exit'
        if question.lower() == "exit":
            print("Exiting program.")
            break

        # Refine the prompt using LLM to detect sensitive or modifying queries
        if not refine_prompt(
            llm_groq, question
        ):  # Pass the selected LLM for refinement
            continue  # Prompt needs refinement, ask the user to refine it

        # Generate the SQL query based on user input
        try:
            response = chain.invoke({"question": question})

            # Extract the SQL query from the response
            sql_query = response.split("SQLQuery:")[-1].strip()
        except Exception as e:
            print(f"Warning: An error occurred while generating the SQL query: {e}")
            continue  # Skip to the next input

        # Execute the SQL query using the database connection
        try:
            result = db.run(sql_query)
        except Exception as e:
            print(f"Warning: An error occurred while executing the SQL query: {e}")
            continue  # Skip to the next input

        # Allow the user to switch between LLMs
        llm_choice = input("Choose LLM (groq/llama2): ").strip().lower()
        if llm_choice == "groq":
            llm = llm_groq
        else:
            print("Invalid LLM choice. Using default 'groq'.")
            llm = llm_groq

        # Check if the user requested a visualization
        try:
            if check_for_visualization_request(llm_groq, question):
                print("Visualization requested. Generating chart...")
                # Ask the selected LLM to generate Python code for the visualization
                generated_code = ask_llm_to_generate_code(llm_groq, sql_query, result)
                print(f"Generated Python code:\n{generated_code}")

                # Execute the generated Python code to create and save the visualization
                execute_generated_code(generated_code)
            else:
                print(f"Answer: The SQL result is {result}")
        except Exception as e:
            print(f"Warning: An error occurred during visualization: {e}")
            continue  # Skip to the next input

        # Fill the answer template with the question, query, and result
        try:
            filled_prompt = answer_prompt.format(
                question=question, query=sql_query, result=result
            )

            # Create the message input for the LLM
            human_message = HumanMessage(content=filled_prompt)

            # Generate the final answer using the LLM
            final_response = llm.invoke([human_message])

            # Print the final result
            print(final_response.content)
        except Exception as e:
            print(f"Warning: An error occurred while generating the final answer: {e}")
            continue  # Skip to the next input

    except KeyboardInterrupt:
        print("\nProgram interrupted. Please use 'exit' to quit next time.")
    except Exception as e:
        print(f"Unexpected error occurred: {e}. Continuing the program...")
