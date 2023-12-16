import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the CSV file into a DataFrame
file_path = 'data/telecom_cleaned.csv'  
df = pd.read_csv(file_path)

@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)


# Calculate throughput for Downlink (DL) and Uplink (UL) in kbps
df['Throughput DL (kbps)'] = (df['Total DL (Bytes)'] * 8) / (df['Dur. (ms)'] / 1000) / 1024
df['Throughput UL (kbps)'] = (df['Total UL (Bytes)'] * 8) / (df['Dur. (ms)'] / 1000) / 1024
df['Throughput'] = df['Throughput DL (kbps)'] + df['Throughput UL (kbps)']


# Sidebar for user interaction
st.sidebar.header('User Interaction')

def display_top_handsets(df, n=10):
    handset_counts = df['Handset Type'].value_counts().head(n)
    st.write(f"Top {n} Handsets Used by Customers:")
    
    # Plotting the top handsets
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=handset_counts.values, y=handset_counts.index, palette='viridis', ax=ax)
    ax.set_xlabel('Number of Users')
    ax.set_ylabel('Handset Types')
    ax.set_title(f'Top {n} Handsets Used by Customers')
    st.pyplot(fig)

def display_top_manufacturers(df, n=3):
    manufacturer_counts = df['Handset Manufacturer'].value_counts().head(n)
    st.write(f"Top {n} Handset Manufacturers:")
    
    # Plotting the top manufacturers
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=manufacturer_counts.values, y=manufacturer_counts.index, palette='viridis', ax=ax)
    ax.set_xlabel('Number of Users')
    ax.set_ylabel('Handset Manufacturers')
    ax.set_title(f'Top {n} Handset Manufacturers')
    st.pyplot(fig)

def display_top_handsets_per_manufacturer(df, selected_manufacturer, top_handsets=5):
    st.write(f"Top {top_handsets} Handsets for {selected_manufacturer}:")
    
    # To Filter data for the selected manufacturer
    manufacturer_df = df[df['Handset Manufacturer'] == selected_manufacturer]
    
    # To Get the top N (as as much as selected on the slider)handsets for the selected manufacturer
    top_handset_counts = manufacturer_df['Handset Type'].value_counts().head(top_handsets)
    
    # Plotting the top handsets for the selected manufacturer
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_handset_counts.values, y=top_handset_counts.index, palette='viridis', ax=ax)
    ax.set_xlabel('Number of Users')
    ax.set_ylabel('Handset Types')
    ax.set_title(f'Top {top_handsets} Handsets for {selected_manufacturer}')
    st.pyplot(fig)

def display_aggregated_xdr_sessions_per_user(df):
    sessions_per_user = df.groupby('MSISDN/Number')['Bearer Id'].count().reset_index()
    sessions_per_user.rename(columns={'Bearer Id': 'XDR Sessions'}, inplace=True)
    st.header("XDR Sessions per User:")
    st.write(sessions_per_user)

def display_aggregated_session_duration_per_user(df):
    sessions_per_user = df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index()
    sessions_per_user.rename(columns={'Dur. (ms)': 'Total Session Duration (ms)'}, inplace=True)
    sessions_per_user = sessions_per_user.sort_values(by='Total Session Duration (ms)', ascending=False)
    
    st.header("Session Duration per User:")
    st.write(sessions_per_user)
# To display  Total Download and Upload Data per User
def display_aggregated_total_data_per_user(df):
    total_data_per_user = df.groupby('MSISDN/Number')[['Total DL (Bytes)', 'Total UL (Bytes)']].sum().reset_index()
    st.header("Total Download and Upload Data per User:")
    st.write(total_data_per_user)

def total_data_volume_per_application(df, user_identifier='MSISDN/Number'):
    st.title("Total Data Volume per Application")

    # List of columns representing different applications
    application_columns = [
        'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
        'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)',
        'Other DL (Bytes)'
    ]

    # Create a new DataFrame to store aggregated results
    aggregated_df = pd.DataFrame()

    # Aggregate total volume for each application per user
    for column in application_columns:
        # Group by user and sum the volume for the current application
        app_aggregated = df.groupby(user_identifier)[column].sum().reset_index()

        # Rename the column to include the application name
        app_aggregated.rename(columns={column: f'Total {column}'}, inplace=True)

        # Merge with the aggregated_df
        if aggregated_df.empty:
            aggregated_df = app_aggregated
        else:
            aggregated_df = pd.merge(aggregated_df, app_aggregated, on=user_identifier)

    # Display the aggregated results
    st.subheader("Aggregated Data:")
    st.dataframe(aggregated_df)

def calculate_dispersion_parameters(df, quantitative_variables):
    dispersion_data = []

    for variable in quantitative_variables:
        data = df[variable].dropna()  # Remove missing values if any

        # Calculate dispersion parameters
        range_value = data.max() - data.min()
        iqr_value = data.quantile(0.75) - data.quantile(0.25)
        variance_value = data.var()
        std_deviation_value = data.std()
        cv_value = (std_deviation_value / data.mean()) * 100

        # Store results in a dictionary
        dispersion_data.append({
            'Variable': variable,
            'Range': range_value,
            'IQR': iqr_value,
            'Variance': variance_value,
            'Standard Deviation': std_deviation_value,
            'Coefficient of Variation': cv_value
        })

    # Create a DataFrame from the results
    dispersion_df = pd.DataFrame(dispersion_data)
    return dispersion_df

def streamlit_dispersion_analysis(df):
    st.title("Non-Graphical Univariate Analysis - Dispersion Parameters")

    # Sidebar for user interaction
    st.sidebar.header('Select Quantitative Variables')
    quantitative_variables = st.sidebar.multiselect("Select Variables", df.select_dtypes(include='number').columns)

    if not quantitative_variables:
        st.warning("Please select at least one quantitative variable.")
        return

    # Calculate dispersion parameters
    dispersion_df = calculate_dispersion_parameters(df, quantitative_variables)

    # Display the results
    st.subheader("Dispersion Parameters for Selected Variables:")
    st.dataframe(dispersion_df)

def graphical_univariate_analysis(df):
    st.title("Graphical Univariate Analysis")

    # Sidebar for user interaction
    st.sidebar.header('Select Variable and Plot Type')
    variable = st.sidebar.selectbox("Select Variable", df.columns)
    plot_type = st.sidebar.radio("Select Plot Type", ['Histogram', 'Box Plot', 'Bar Plot'])

    # Main plot
    st.subheader(f"{plot_type} for {variable}")

    if plot_type == 'Histogram':
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[variable], kde=True, ax=ax)
        st.pyplot(fig)
    elif plot_type == 'Box Plot':
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=df[variable], ax=ax)
        st.pyplot(fig)
    elif plot_type == 'Bar Plot':
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x=df[variable], ax=ax)
        st.pyplot(fig)


def bivariate_analysis(df, user_identifier='MSISDN/Number'):
    st.title("Bivariate Analysis: Application vs. Total DL+UL Data")

    # Sidebar for user interaction
    st.sidebar.header('Select Application')
    application = st.sidebar.selectbox("Select Application", df.filter(like='DL (Bytes)').columns)

    # Main plot
    st.subheader(f"Relationship between {application} and Total DL+UL Data")

    # Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=df[application], y=df['Total DL (Bytes)'] + df['Total UL (Bytes)'], ax=ax)
    ax.set_xlabel(f"{application} Data")
    ax.set_ylabel("Total DL+UL Data")
    ax.set_title(f"Scatter Plot: {application} vs. Total DL+UL Data")
    st.pyplot(fig)

    # Correlation Coefficient
    correlation_coefficient = df[application].corr(df['Total DL (Bytes)'] + df['Total UL (Bytes)'])
    st.subheader(f"Correlation Coefficient: {correlation_coefficient:.4f}")

    # Heatmap
    st.subheader("Correlation Heatmap")
    correlation_matrix = df.filter(like='DL (Bytes)').corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

def variable_transformations(df, user_identifier='MSISDN/Number'):
    st.title("Variable Transformations")

    # Compute total duration per user
    total_duration_per_user = df.groupby(user_identifier)['Dur. (ms)'].sum().reset_index()
    total_duration_per_user.rename(columns={'Dur. (ms)': 'Total Duration (ms)'}, inplace=True)

    # Create deciles based on total duration
    total_duration_per_user['Decile'] = pd.qcut(total_duration_per_user['Total Duration (ms)'], q=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=False, duplicates='drop')

    # Display the table with total duration and deciles
    st.subheader("Total Duration and Deciles")
    st.dataframe(total_duration_per_user)

    # Compute total data (DL+UL) per decile class
    df_with_deciles = pd.merge(df, total_duration_per_user[[user_identifier, 'Decile']], on=user_identifier)

    total_data_per_decile = df_with_deciles.groupby('Decile')[['Total DL (Bytes)', 'Total UL (Bytes)']].sum().reset_index()

    # Display the table with total data per decile class
    st.subheader("Total Data (DL+UL) per Decile Class")
    st.dataframe(total_data_per_decile)

def correlation_analysis(df):
    st.title("Correlation Analysis")

    # Select relevant columns for correlation analysis
    selected_columns = [
        'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
        'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)',
        'Other DL (Bytes)'
    ]

    # Subset the DataFrame with selected columns
    correlation_data = df[selected_columns]

    # Compute the correlation matrix
    correlation_matrix = correlation_data.corr()

    # Display the correlation matrix
    st.subheader("Correlation Matrix")
    st.dataframe(correlation_matrix)

    # Interpretation
    st.subheader("Interpretation")
    st.markdown("""
    - A correlation close to 1 indicates a strong positive correlation, meaning if one variable increases, the other tends to increase as well.
    - A correlation close to -1 indicates a strong negative correlation, meaning if one variable increases, the other tends to decrease.
    - A correlation close to 0 indicates a weak or no linear correlation between the variables.
    """)

def dimensionality_reduction(df):
    st.title("Dimensionality Reduction - Principal Component Analysis (PCA)")

    # Select relevant columns for PCA
    selected_columns = [
        'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
        'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)',
        'Other DL (Bytes)'
    ]

    # Subset the DataFrame with selected columns
    pca_data = df[selected_columns]

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pca_data)

    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)

    # Display explained variance ratio
    st.subheader("Explained Variance Ratio")
    st.bar_chart(pca.explained_variance_ratio_)

    # Interpretation
    st.subheader("Interpretation")
    st.markdown("""
    - The explained variance ratio represents the proportion of the dataset's variance captured by each principal component.
    - A higher explained variance ratio for a principal component indicates that it retains more information from the original data.
    - You can choose the number of principal components based on the cumulative explained variance to achieve a desired level of dimensionality reduction.
    - The bar chart above helps visualize the contribution of each principal component to the overall variance.
    """)

def display_average_tcp_retransmission(df):
    avg_tcp_retransmission = df.groupby('MSISDN/Number')['TCP DL Retrans. Vol (Bytes)'].mean().reset_index()
    st.header("Average TCP Retransmission per Customer:")
    st.write(avg_tcp_retransmission)

def display_average_rtt(df):
    avg_rtt = df.groupby('MSISDN/Number')['Avg RTT DL (ms)'].mean().reset_index()
    st.header("Average RTT per Customer:")
    st.write(avg_rtt)
def display_aggregate_handset_types(df, user_identifier='MSISDN/Number'):
    # Check if 'Handset Type' column exists in the DataFrame
    if 'Handset Type' not in df.columns:
        st.warning("Column 'Handset Type' not found in the dataset.")
        return None

    # Group by the specified user identifier and count the unique handset types
    handset_counts = df.groupby(user_identifier)['Handset Type'].value_counts().unstack().reset_index()
    handset_counts = handset_counts.fillna(0)  # Fill NaN values with 0 for better presentation

    # Generate a unique key for the slider widget
    slider_key = f"slider_{user_identifier}_handset_{id(handset_counts)}"

    # Display only a subset of the data based on the user's choice
    st.header("Aggregate Handset Types per Customer:")
    rows_to_display = st.slider(f'Select number of rows to display:', min_value=1, max_value=len(handset_counts), value=10, key=slider_key)
    st.table(handset_counts.head(rows_to_display))

def display_average_throughput(df):
    avg_throughput = df.groupby('MSISDN/Number')['Throughput'].mean().reset_index()
    st.header("Average Throughput per Customer:")
    st.write(avg_throughput)

def list_top_values(df, column_name, top_n=10):
    top_values = df[column_name].value_counts().head(top_n).reset_index()
    top_values.columns = [column_name, 'Count']
    return top_values
def list_top_bottom_most_frequent_values(df, column_name, n=10):
    # Check if the specified column exists in the DataFrame
    if column_name not in df.columns:
        st.warning(f"Column '{column_name}' not found in the dataset.")
        return None

    # Compute the most frequent values
    most_frequent_values = df[column_name].value_counts().head(n).reset_index()
    most_frequent_values.columns = [column_name, 'Count']

    # Compute the least frequent values
    least_frequent_values = df[column_name].value_counts().tail(n).reset_index()
    least_frequent_values.columns = [column_name, 'Count']

    # Compute the top values
    top_values = df.nlargest(n, column_name)
    
    # Compute the bottom values
    bottom_values = df.nsmallest(n, column_name)

    # Display the results
    st.subheader(f"Top {n} {column_name} Values:")
    st.dataframe(top_values)

    st.subheader(f"Bottom {n} {column_name} Values:")
    st.dataframe(bottom_values)

    st.subheader(f"Most Frequent {column_name} Values:")
    st.dataframe(most_frequent_values)

def display_top_average_throughput_per_handset(df, top_n=20):
   
    # Group by 'Handset Type' and calculate the average throughput
    average_throughput_per_handset = df.groupby('Handset Type')['Throughput'].mean().reset_index()

    # Select the top N handsets based on average throughput
    top_handsets = average_throughput_per_handset.nlargest(top_n, 'Throughput')

    # Display the distribution using a boxplot
    st.title(f"Distribution of Top {top_n} Average Throughput per Handset Type")
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Throughput', y='Handset Type', data=df[df['Handset Type'].isin(top_handsets['Handset Type'])],hue='Handset Type', legend=False, palette='viridis')
    # sns.boxplot(x='Throughput', y='Handset Type', data=df, hue='Handset Type', legend=False, palette='viridis')
    plt.xlabel('Average Throughput (kbps)')
    plt.ylabel('Handset Type')
    plt.title(f'Distribution of Top {top_n} Average Throughput per Handset Type')
    st.pyplot(plt)

    # Display the DataFrame with top average throughput per handset type
    st.subheader(f"Top {top_n} Average Throughput per Handset Type:")
    st.dataframe(top_handsets)


    """This function will display a boxplot showing the distribution 
    of average throughput per handset type and also provide a 
    DataFrame with the average throughput values."""

def analyze_tcp_retransmission_per_handset(df, top_n=20):
    """
    Analyze and display average TCP retransmission for each handset type.
    Display a boxplot showing the distribution of average TCP retransmission per handset type.
    Provide a DataFrame with average TCP retransmission values.
    """

    if 'TCP DL Retrans. Vol (Bytes)' not in df.columns:
        st.warning("Column 'TCP DL Retrans. Vol (Bytes)' not found in the dataset.")
        return

    # Group by 'Handset Type' and calculate the average TCP retransmission
    avg_tcp_retransmission_per_handset = df.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean().reset_index()

    # Sort by average TCP retransmission in descending order
    avg_tcp_retransmission_per_handset = avg_tcp_retransmission_per_handset.sort_values(by='TCP DL Retrans. Vol (Bytes)', ascending=False)

    # Display the top N handsets with the highest average TCP retransmission
    st.title(f"Top {top_n} Handsets with Highest Average TCP Retransmission")
    st.dataframe(avg_tcp_retransmission_per_handset.head(top_n))

    # Display the distribution using a barplot
    st.title("Average TCP Retransmission per Handset Type")
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 10))
    sns.barplot(x='TCP DL Retrans. Vol (Bytes)', y='Handset Type', data=avg_tcp_retransmission_per_handset.head(top_n), palette='viridis')
    plt.xlabel('Average TCP Retransmission Volume (Bytes)')
    plt.ylabel('Handset Type')
    plt.title('Average TCP Retransmission per Handset Type')
    st.pyplot(plt)

def kmeans_clustering_analysis(df):
    st.title("K-Means Clustering Analysis")

    # Select relevant numerical features for clustering
    selected_features = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']

    # Filter DataFrame to include only selected features
    df_clustering = df[selected_features].dropna()

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clustering)

    # Perform k-means clustering with k=3
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_clustering['Cluster'] = kmeans.fit_predict(scaled_data)

    # Provide a brief description of each cluster
    cluster_descriptions = {
        0: "Low RTT, High Bearer Throughput (DL and UL)",
        1: "Medium RTT, Medium Bearer Throughput (DL and UL)",
        2: "High RTT, Low Bearer Throughput (DL and UL)"
    }

    # Map cluster labels to descriptions
    df_clustering['Cluster Description'] = df_clustering['Cluster'].map(cluster_descriptions)

    # Display the DataFrame with cluster assignments and descriptions
    st.subheader("Cluster Assignments and Descriptions:")
    st.dataframe(df_clustering[['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 'Cluster', 'Cluster Description']])

    # Display a countplot of cluster distribution
    st.subheader("Distribution of Clusters:")
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Cluster', data=df_clustering, palette='viridis')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title('Distribution of Clusters')
    st.pyplot(plt)

def calculate_engagement_scores(df):
    st.title("Calculate Engagement Scores")

    # Select relevant numerical features for clustering
    selected_features = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']

    # Filter DataFrame to include only selected features
    df_clustering = df[selected_features].dropna()

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clustering)

    # Perform k-means clustering with k=3 (assuming you have already done this)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_clustering['Cluster'] = kmeans.fit_predict(scaled_data)

    # Find the less engaged cluster (assumes 0, 1, and 2 are cluster labels)
    less_engaged_cluster_label = df_clustering['Cluster'].value_counts().idxmin()

    # Extract data points of the less engaged cluster
    less_engaged_cluster_data = df_clustering[df_clustering['Cluster'] == less_engaged_cluster_label][selected_features]

    # Calculate the centroid of the less engaged cluster
    centroid_less_engaged_cluster = less_engaged_cluster_data.mean().values.reshape(1, -1)

    # Calculate the Euclidean distance between each user and the centroid of the less engaged cluster
    engagement_scores = pairwise_distances_argmin_min(scaled_data, centroid_less_engaged_cluster)[1]

    # Add the engagement scores to the original DataFrame
    df['Engagement Score'] = engagement_scores

    # Display the DataFrame with engagement scores
    st.subheader("Engagement Scores:")
    st.dataframe(df[['MSISDN/Number', 'Engagement Score']])

def calculate_experience_scores(df):
    st.title("Calculate Experience Scores")

    # Select relevant numerical features for clustering
    selected_features = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']

    # Filter DataFrame to include only selected features
    df_clustering = df[selected_features].dropna()

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clustering)

    # Perform k-means clustering with k=3 
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_clustering['Cluster'] = kmeans.fit_predict(scaled_data)

    # Find the cluster associated with the worst experience (assumes 0, 1, and 2 are cluster labels)
    worst_experience_cluster_label = df_clustering['Cluster'].value_counts().idxmax()

    # Extract data points of the worst experience cluster
    worst_experience_cluster_data = df_clustering[df_clustering['Cluster'] == worst_experience_cluster_label][selected_features]

    # Calculate the centroid of the worst experience cluster
    centroid_worst_experience_cluster = worst_experience_cluster_data.mean().values.reshape(1, -1)

    # Calculate the Euclidean distance between each user and the centroid of the worst experience cluster
    experience_scores = pairwise_distances_argmin_min(scaled_data, centroid_worst_experience_cluster)[1]

    # Add the experience scores to the original DataFrame
    df['Experience Score'] = experience_scores

    # Display the DataFrame with experience scores
    st.subheader("Experience Scores:")
    st.dataframe(df[['MSISDN/Number', 'Experience Score']])


def user_experience_analysis(df):
    st.title("User Experience Analysis")

    st.subheader("Dataset")
    st.dataframe(df)

    display_average_tcp_retransmission(df)
    display_average_rtt(df)
    display_aggregate_handset_types(df)
    display_average_throughput(df)

    st.subheader("Top Values")
    top_tcp_values = list_top_values(df, 'TCP DL Retrans. Vol (Bytes)')
    top_rtt_values = list_top_values(df, 'Avg RTT DL (ms)')
    top_throughput_values = list_top_values(df, 'Throughput')

    st.write("Top TCP Values:")
    st.write(top_tcp_values)

    st.write("Top RTT Values:")
    st.write(top_rtt_values)

    st.write("Top Throughput Values:")
    st.write(top_throughput_values)

def calculate_satisfaction_scores(df):
    st.title("Calculate Satisfaction Scores")

    selected_features = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']

    df_clustering = df[selected_features].dropna()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clustering)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_clustering['Cluster'] = kmeans.fit_predict(scaled_data)

    less_engaged_cluster_label = df_clustering['Cluster'].value_counts().idxmin()
    less_engaged_cluster_data = df_clustering[df_clustering['Cluster'] == less_engaged_cluster_label][selected_features]

    centroid_less_engaged_cluster = less_engaged_cluster_data.mean().values.reshape(1, -1)
    engagement_scores = pairwise_distances_argmin_min(scaled_data, centroid_less_engaged_cluster)[1]

    worst_experience_cluster_label = df_clustering['Cluster'].value_counts().idxmax()
    worst_experience_cluster_data = df_clustering[df_clustering['Cluster'] == worst_experience_cluster_label][selected_features]

    centroid_worst_experience_cluster = worst_experience_cluster_data.mean().values.reshape(1, -1)
    experience_scores = pairwise_distances_argmin_min(scaled_data, centroid_worst_experience_cluster)[1]

    satisfaction_scores = (engagement_scores + experience_scores) / 2

    df['Satisfaction Score'] = satisfaction_scores

    top_satisfied_customers = df.nlargest(10, 'Satisfaction Score')[['MSISDN/Number', 'Satisfaction Score']]
    st.subheader("Top 10 Satisfied Customers:")
    st.dataframe(top_satisfied_customers)

def build_regression_model(df):
    st.title("Build Regression Model for Satisfaction Prediction")

    features = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']
    target = 'Satisfaction Score'

    df_regression = df[features + [target]].dropna()

    X_train, X_test, y_train, y_test = train_test_split(df_regression[features], df_regression[target], test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation Metrics:")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"R-squared (R2): {r2}")

    st.subheader("Sample Predictions vs. Actual:")
    sample_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    st.dataframe(sample_predictions.head(10))

def kmeans_clustering(df):
    st.title("K-Means Clustering on Engagement & Experience Scores")

    features = ['Engagement Score', 'Experience Score', 'Satisfaction Score']
    df_clustering = df[features].dropna()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clustering)

    kmeans = KMeans(n_clusters=2, random_state=42)
    df_clustering['Cluster'] = kmeans.fit_predict(scaled_data)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Engagement Score', y='Experience Score', hue='Cluster', data=df_clustering, palette='viridis', ax=ax)
    ax.set_title('K-Means Clustering (k=2) on Engagement & Experience Scores')
    st.pyplot(fig)

    st.subheader("Cluster Information:")
    cluster_info = df_clustering.groupby('Cluster').mean()
    st.dataframe(cluster_info)

    avg_scores_per_cluster = df_clustering.groupby('Cluster')[['Satisfaction Score', 'Experience Score']].mean()
    st.subheader("Average Satisfaction & Experience Scores per Cluster:")
    st.dataframe(avg_scores_per_cluster)

if __name__ == "__main__":
    st.title('Exploratory Data Analysis (EDA) App')
    user_identifier = 'MSISDN/Number'

    num_top_handsets = st.sidebar.selectbox("Select Number of Top Handsets Used", range(1, df['Handset Type'].nunique() + 1), 9)
    num_top_manufacturers = st.sidebar.selectbox("Select Number of Top Handset Manufacturers", range(1, df['Handset Manufacturer'].nunique() + 1), 2)

    manufacturer_counts = df['Handset Manufacturer'].value_counts().head(num_top_manufacturers)
    sorted_manufacturers = manufacturer_counts.index.tolist()
    selected_manufacturer = st.sidebar.selectbox("Select Handset Manufacturer", sorted_manufacturers)

    display_top_handsets(df.head(), n=num_top_handsets)
    display_top_manufacturers(df, n=num_top_manufacturers)
    display_top_handsets_per_manufacturer(df, selected_manufacturer, top_handsets=5)
    display_aggregated_xdr_sessions_per_user(df)
    display_aggregated_session_duration_per_user(df)
    display_aggregated_total_data_per_user(df)

    st.title('Data Volume Analysis')
    total_data_volume_per_application(df, user_identifier)
    streamlit_dispersion_analysis(df)
    graphical_univariate_analysis(df)
    bivariate_analysis(df)
    variable_transformations(df)
    correlation_analysis(df)
    dimensionality_reduction(df)

    user_experience_analysis(df)

    st.title("Top, Bottom, and Most Frequent Values Analysis")
    columns_to_analyze = ['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Throughput']
    for column in columns_to_analyze:
        st.header(f"Analysis for {column}")
        list_top_bottom_most_frequent_values(df, column, n=10)

    st.title("Top Average Throughput Analysis")
    top_n_handsets = st.sidebar.slider('Select Number of Top Handsets to Display:', min_value=1, max_value=df['Handset Type'].nunique(), value=20)
    display_top_average_throughput_per_handset(df, top_n=top_n_handsets)

    st.title("TCP Retransmission Analysis per Handset Type")
    analyze_tcp_retransmission_per_handset(df, top_n=20)

    st.title("User Experience Analysis with K-Means Clustering")
    kmeans_clustering_analysis(df)

    st.title("Engagement Score Calculation App")
    calculate_engagement_scores(df)

    st.title("Experience Score Calculation App")
    calculate_experience_scores(df)

    st.title("Satisfaction Score Calculation App")
    calculate_satisfaction_scores(df)

    st.title("Regression Model Building App")
    build_regression_model(df)

    st.title("K-Means Clustering App")
    kmeans_clustering(df)
