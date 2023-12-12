import streamlit as st
import pandas as pd
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

# Function to fetch data from PostgreSQL
def fetch_data():
    connection = connect_to_db()
    cursor = connection.cursor()

    # Replace 'your_table_name' with your actual table name
    cursor.execute("SELECT * FROM xdr_data")
    data = cursor.fetchall()

    connection.close()

    return data

# Main Streamlit app
def main():
    st.title("PostgreSQL and Streamlit Dashboard")

    # Fetch data from PostgreSQL
    data = fetch_data()

    # Convert data to a DataFrame for further analysis
    df = pd.DataFrame(data, columns=["id","Bearer Id", "Start", "Start ms", "End", "End ms","Dur. (ms)","IMSI", "MSISDN/Number", 
     "IMEI","Last Location Name", "Avg RTT DL (ms)","Avg RTT UL (ms)" , "Avg Bearer TP DL (kbps)" , "Avg Bearer TP UL (kbps)", 
     "TCP DL Retrans. Vol (Bytes)" , "TCP UL Retrans. Vol (Bytes)" , "DL TP < 50 Kbps (%)" , "50 Kbps < DL TP < 250 Kbps (%)" , 
      "250 Kbps < DL TP < 1 Mbps (%)" , "DL TP > 1 Mbps (%)" ,  "UL TP < 10 Kbps (%)",  "10 Kbps < UL TP < 50 Kbps (%)", 
      "50 Kbps < UL TP < 300 Kbps (%)" , "UL TP > 300 Kbps (%)", "HTTP DL (Bytes)", "HTTP UL (Bytes)" ,"Activity Duration DL (ms)", 
      "Activity Duration UL (ms)", "Dur. (ms).1", "Handset Manufacturer" ,"Handset Type" , "Nb of sec with 125000B < Vol DL", 
      "Nb of sec with 1250B < Vol UL < 6250B","Nb of sec with 31250B < Vol DL < 125000B" ,"Nb of sec with 37500B < Vol UL",
      "Nb of sec with 6250B < Vol DL < 31250B", "Nb of sec with 6250B < Vol UL < 37500B", "Nb of sec with Vol DL < 6250B",
       "Nb of sec with Vol UL < 1250B" , "Social Media DL (Bytes)","Social Media UL (Bytes)" ,"Google DL (Bytes)",   "Google UL (Bytes)",
       "Email DL (Bytes)", "Email UL (Bytes)", "Youtube DL (Bytes)",  "Youtube UL (Bytes)", "Netflix DL (Bytes)",
        "Netflix UL (Bytes)" ,"Gaming DL (Bytes)" ,
         "Gaming UL (Bytes)", "Other DL (Bytes)", "Other UL (Bytes)","Total UL (Bytes)", "Total DL (Bytes)"
    ])  # Adjust column names accordingly

    # Display raw data in a Streamlit table
    st.subheader("Raw Data")
    # st.table(df)
    st.write(df.head())

    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Display a line chart using Plotly
    st.subheader("Line Chart")

    # Allow user to select columns for the line chart
    selected_columns = st.multiselect("Select columns for the Line Chart", df.columns[1:])  # Exclude the first column (id) if it's not relevant
    if selected_columns:
        st.line_chart(df[selected_columns])

    # Create a sidebar for user input
    st.sidebar.header("User Input")

    # Example: Allow user to filter data based on a specific column
    selected_column = st.sidebar.selectbox("Select a Column", df.columns[1:])  # Exclude the first column (id) if it's not relevant
    filtered_data = df[selected_column]

    # Display filtered data
    st.subheader("Filtered Data")
    st.write(filtered_data)

if __name__ == "__main__":
    main()
