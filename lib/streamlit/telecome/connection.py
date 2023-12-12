import streamlit as st
import psycopg2

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