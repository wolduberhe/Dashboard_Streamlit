import streamlit as st
import psycopg2
import pandas as pd

# Function to connect to PostgreSQL
def connect_to_db():
    connection = psycopg2.connect(
        dbname="telecome",
        user="mebmeressa",
        password="",
        host="localhost",
        port=5432,  # Default is usually 5432
    )
    return connection

# Function to fetch data from PostgreSQL and return a Pandas DataFrame
def fetch_data():
    try:
        connection = connect_to_db()
        cursor = connection.cursor()

        cursor.execute("SELECT * FROM xdr_data")
        data = cursor.fetchall()

        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=columns)

        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
    finally:
        if connection:
            connection.close()

# Main Streamlit app
def main():
    st.title("PostgreSQL and Streamlit Example")

    # Fetch data from PostgreSQL as a Pandas DataFrame
    data = fetch_data()

    # Display data in a Streamlit table
    
    st.write(data.head(15))
    st.table(data.head(100))

if __name__ == "__main__":
    main()
