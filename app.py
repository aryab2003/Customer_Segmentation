import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import metrics
import streamlit as st
import io


# Load data
@st.cache_data  # Cache data for faster reloading
def load_data():
    return pd.read_csv("Mall_Customers.csv")


customer_data = load_data()

# Sidebar for selecting features and number of clusters
st.sidebar.title("Customer Segmentation")
# Exclude "Gender" column from selectable features
selected_features = st.sidebar.multiselect(
    "Select Features", customer_data.columns.drop("Gender")
)
k_clusters = st.sidebar.slider("Number of Clusters", 1, 10, 5)

if len(selected_features) >= 2:
    # Data Summary
    st.subheader("Data Summary")
    st.write(customer_data[selected_features].describe())
    st.write(customer_data[selected_features].info())

    # Data Visualization
    st.subheader("Data Visualization")
    # Add interactive visualizations using seaborn or other libraries
    # For example:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=selected_features[0], y=selected_features[1], data=customer_data, ax=ax
    )
    st.pyplot(fig)

    # Perform K-means clustering
    X = customer_data[selected_features].values
    kmeans = KMeans(n_clusters=k_clusters, init="k-means++", random_state=0)
    Y = kmeans.fit_predict(X)

    # Plot clusters
    fig, ax = plt.subplots(figsize=(8, 8))
    for cluster_label in range(k_clusters):
        plt.scatter(
            X[Y == cluster_label, 0],
            X[Y == cluster_label, 1],
            s=50,
            label=f"Cluster {cluster_label+1}",
        )
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=100,
        c="cyan",
        label="Centroid",
    )
    plt.title("Customer Segmentation")
    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])
    plt.legend()
    st.pyplot(fig)

    # Cluster Analysis
    st.subheader("Cluster Analysis")
    cluster_counts = np.bincount(Y)
    for i in range(k_clusters):
        st.write(f"Cluster {i+1}: {cluster_counts[i]} customers")

    # Model Evaluation (Optional)
    st.subheader("Model Evaluation")
    silhouette_score = metrics.silhouette_score(X, Y)
    st.write(f"Silhouette Score/Clustering Accuracy: {silhouette_score}")

    # Export Results (Optional)
    st.subheader("Export Results")
    export_format = st.selectbox("Select Export Format", ["CSV", "Excel"])
    if export_format == "CSV":
        csv = customer_data[selected_features].to_csv(index=False)
        st.download_button(
            "Download CSV", io.BytesIO(csv.encode()), "customer_data.csv", "text/csv"
        )
    elif export_format == "Excel":
        excel_buffer = io.BytesIO()
        excel_writer = pd.ExcelWriter(excel_buffer, engine="xlsxwriter")
        customer_data[selected_features].to_excel(excel_writer, index=False)
        excel_writer.close()  # Close the Excel writer to save the buffer content
        excel_buffer.seek(0)
        st.download_button(
            "Download Excel",
            excel_buffer,
            file_name="customer_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # Elbow Curve
    st.subheader("Elbow Curve")
    sns.set()  # Set Seaborn style
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    plt.plot(range(1, 11), wcss)
    plt.title("Elbow Curve")
    plt.xlabel("No of clusters")
    plt.ylabel("WCSS value")
    st.pyplot(fig)

else:
    st.write("Please select at least two features to create a scatter plot.")
