# Streamlit BigQuery Query Optimizer

A web application built with Streamlit that provides a user-friendly interface for analyzing and optimizing BigQuery SQL queries using OpenAI's API. This tool helps to improve query performance by offering AI-driven recommendations and generating optimized SQL versions.

## Features

- **AI-Powered Analysis**: Uses OpenAI to analyze BigQuery queries for performance and efficiency.
- **Query Optimization**: Suggests improvements for table scanning, joins, filtering, partitioning, and clustering.
- **GCP Integration**: Supports persistent authentication with Google Cloud for BigQuery API access.
- **Cost Estimation**: Provides an estimate of the cost associated with executing the query.
- **Performance Tracking**: Logs query performance metrics and tracks historical improvements.

## Requirements

- Python 3.8+
- Dependencies:
    - `streamlit`
    - `google-cloud-bigquery`
    - `google-auth`
    - `openai`
    - `pandas`
    - `plotly`
    - `typing_extensions`
    - `uuid`
    - `python-dotenv`
    - `google-cloud-bigquery[pandas]`

To install the required packages, you can use:
```bash
pip install -r requirements.txt
```

## Setup
**1. Google Cloud Authentication:**

- Ensure you have gcloud CLI set up and authenticated on your machine.

- Run the following command if needed:

```bash
gcloud auth application-default login
```

- Make sure the application_default_credentials.json file is present in ~/AppData/Roaming/gcloud/ (or equivalent path for non-Windows systems).

**2. OpenAI API Key:**

- Obtain an API key from OpenAI.

- You’ll be prompted to enter this in the app’s sidebar.

**3. Create a DataSet and Table for Storing Historic Performance**

- Run the following query in the BigQuery

```bash
CREATE TABLE query_optimization.performance_metrics (
    timestamp TIMESTAMP,
    query_id STRING,
    original_bytes INT64,
    optimized_bytes INT64,
    improvement_percentage FLOAT64
);

```

**4. Run the Application:**

```bash
streamlit run app.py
```

## Usage

**1. Analyze Query:** Enter a BigQuery SQL query and click the "Analyze Query" button. The app will display:

- Processed bytes
- Estimated query cost
- AI recommendations for optimization

**2. Optimize Query:** Click the "Optimize Query" button to generate an improved SQL query based on AI analysis. The app displays:

- Original vs. optimized query cost
- Percentage improvement in query performance
- Optimized SQL query

**3. Historical Performance:** View historical data on query improvements with visual charts, giving insights into performance trends.

## Project Structure

- *app.py* : Main application code
- *BigQueryOptimizer class* : Core logic for interacting with BigQuery API and optimizing queries
- *initialize_gcp* : Handles Google Cloud authentication
- *create_performance_chart* : Generates visualizations of query performance metrics

## Logging
Logs are saved in query_optimizer.log to track errors, initialization, and performance data.

## Author
 *- Aditya Goswami*

