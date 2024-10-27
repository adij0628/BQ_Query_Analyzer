"""
Streamlit BigQuery Query Optimizer
=================================

A web application built with Streamlit that provides a user-friendly interface
for optimizing BigQuery queries using AI. Features persistent GCP authentication.

Requirements:
Python 3.8+
- streamlit
- google-cloud-bigquery
- google-auth
- openai
- pandas
- plotly
- typing_extensions
- uuid
- python-dotenv
- google-cloud-bigquery[pandas]

Author: Aditya Goswami
Version: 1.2.0
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery
from google.api_core import retry
import google.auth
from openai import OpenAI
from datetime import datetime
import json
import logging
import os
from typing import Dict, List, Optional, Tuple
import uuid


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('query_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@st.cache_resource
def initialize_gcp():
    """
    Initialize GCP authentication and return credentials and project ID.
    This function is cached by Streamlit to prevent repeated authentication.
    """
    try:
        # Authenticate with GCP if not already authenticated
        if not os.path.exists(os.path.expanduser('~/AppData/Roaming/gcloud/application_default_credentials.json')):
            auth_result = os.system("gcloud auth application-default login --quiet")
            if auth_result != 0:
                raise Exception("GCP Authentication failed")
        
        # Get credentials and project ID
        creds, project_id = google.auth.default()
        logger.info(f"Successfully authenticated with GCP project: {project_id}")
        return creds, project_id
        
    except Exception as e:
        logger.error(f"Error during GCP initialization: {str(e)}")
        raise

class BigQueryOptimizer:
    """
    A class to optimize BigQuery queries using AI-powered analysis and suggestions.
    """
    
    def __init__(self, creds, project_id: str, openai_api_key: str):
        """
        Initialize the optimizer with provided credentials.
        
        Args:
            creds: Google Cloud credentials
            project_id (str): Google Cloud project ID
            openai_api_key (str): OpenAI API key for AI-powered optimization
        """
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id, credentials=creds)
        self.openai_client = OpenAI(api_key=openai_api_key)
        logger.info("BigQueryOptimizer initialized successfully")

    def analyze_query(self, query: str) -> Dict:
        """Analyze a BigQuery query using AI."""
        try:
            analysis_prompt = f"""
            Analyze this BigQuery SQL query for optimization opportunities:
            {query}
            
            Consider:
            1. Table scanning efficiency
            2. Join operations
            3. Where clause optimization
            4. Partition and cluster usage
            5. Unnecessary columns
            
            Provide specific recommendations in a structured format.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1
            )
            
            analysis = response.choices[0].message.content
            
            # Dry run to get query statistics
            job_config = bigquery.QueryJobConfig(dry_run=True)
            query_job = self.client.query(query, job_config=job_config)
            
            return {
                "processed_bytes": query_job.total_bytes_processed,
                "ai_analysis": analysis,
                "estimated_cost": self._calculate_cost(query_job.total_bytes_processed)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            raise

    # ... [rest of the BigQueryOptimizer class methods remain the same] ...
    def optimize_query(self, query: str) -> Tuple[str, Dict]:
        """Generate an optimized version of the input query."""
        try:
            analysis = self.analyze_query(query)
            
            optimization_prompt = f"""
            Optimize this BigQuery SQL query based on the analysis:
            Original Query: {query}
            Analysis: {analysis['ai_analysis']}
            
            Return only the optimized SQL query without any explanations.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": optimization_prompt}],
                temperature=0.1
            )
            
            optimized_query = response.choices[0].message.content
            optimized_analysis = self.analyze_query(optimized_query)
            
            return optimized_query, {
                "original_analysis": analysis,
                "optimized_analysis": optimized_analysis,
                "improvement_percentage": self._calculate_improvement(
                    analysis["processed_bytes"],
                    optimized_analysis["processed_bytes"]
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing query: {str(e)}")
            raise

    def _calculate_cost(self, bytes_processed: int) -> float:
        """Calculate estimated query cost."""
        return (bytes_processed / 1099511627776) * 5

    def _calculate_improvement(self, original_bytes: int, optimized_bytes: int) -> float:
        """Calculate percentage improvement."""
        return ((original_bytes - optimized_bytes) / original_bytes) * 100

    def get_performance_history(self) -> pd.DataFrame:
        """Retrieve historical performance metrics."""
        query = f"""
        SELECT *
        FROM `{self.project_id}.query_optimization.performance_metrics`
        ORDER BY timestamp DESC
        LIMIT 1000
        """
        return self.client.query(query).to_dataframe()
    
    def log_performance(self, query_id: str, original_bytes: int, optimized_bytes: int, improvement_percentage: float):
        query = f"""
        INSERT INTO `{self.project_id}.query_optimization.performance_metrics`
        (timestamp, query_id, original_bytes, optimized_bytes, improvement_percentage)
        VALUES (CURRENT_TIMESTAMP(), @query_id, @original_bytes, @optimized_bytes, @improvement_percentage)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("query_id", "STRING", query_id),
                bigquery.ScalarQueryParameter("original_bytes", "INT64", original_bytes),
                bigquery.ScalarQueryParameter("optimized_bytes", "INT64", optimized_bytes),
                bigquery.ScalarQueryParameter("improvement_percentage", "FLOAT64", improvement_percentage),
            ]
        )
        self.client.query(query, job_config=job_config).result()

def create_performance_chart(history_df: pd.DataFrame) -> go.Figure:
    """Create a performance improvement visualization."""
    fig = px.line(
        history_df,
        x='timestamp',
        y='improvement_percentage',
        title='Query Optimization Improvements Over Time'
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Improvement (%)",
        showlegend=True
    )
    return fig

def main():
    st.set_page_config(
        page_title="BigQuery Query Optimizer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🔍 BigQuery Query Optimizer")
    
    # Initialize session state for authentication status
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
        
    # # Sidebar configuration
    st.sidebar.header("Configuration")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", "", type="password")
    
    if not openai_api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar.")
        return
    
    try:
        # Initialize GCP authentication only once
        if not st.session_state.authenticated:
            with st.spinner("Authenticating with Google Cloud..."):
                creds, project_id = initialize_gcp()
                st.session_state.creds = creds
                st.session_state.project_id = project_id
                st.session_state.authenticated = True
        
        # Display connected project information
        st.sidebar.success(f"Connected to GCP Project: {st.session_state.project_id}")
        
        # Initialize optimizer with cached credentials
        optimizer = BigQueryOptimizer(
            st.session_state.creds,
            st.session_state.project_id,
            openai_api_key
        )
        
        # Main query input area
        st.header("Query Input")
        query = st.text_area(
            "Enter your BigQuery SQL query:",
            height=200,
            placeholder="SELECT * FROM `project.dataset.table`..."
        )
        
        col1, col2 = st.columns(2)
        analyze_button = col1.button("Analyze Query")
        optimize_button = col2.button("Optimize Query")
        
        if query and analyze_button:
            with st.spinner("Analyzing query..."):
                try:
                    analysis = optimizer.analyze_query(query)
                    
                    st.header("Analysis Results")
                    
                    # Display metrics in columns
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    metric_col1.metric(
                        "Processed Bytes",
                        f"{analysis['processed_bytes'] / 1e9:.2f} GB"
                    )
                    
                    metric_col2.metric(
                        "Estimated Cost",
                        f"${analysis['estimated_cost']:.4f}"
                    )
                    
                    # Display AI analysis
                    st.subheader("AI Recommendations")
                    st.markdown(analysis['ai_analysis'])
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    logger.error(f"Analysis error: {str(e)}")
        
        if query and optimize_button:
            with st.spinner("Optimizing query..."):
                try:
                    optimized_query, optimization_details = optimizer.optimize_query(query)
                    optimizer.log_performance(
                query_id=str(uuid.uuid4()),  # Generate a unique ID for each query
                original_bytes=optimization_details['original_analysis']['processed_bytes'],
                optimized_bytes=optimization_details['optimized_analysis']['processed_bytes'],
                improvement_percentage=optimization_details['improvement_percentage']
            )  # Generate a unique ID for each query
                    
                    st.header("Optimization Results")
                    
                    # Display improvement metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    improvement = optimization_details['improvement_percentage']
                    metric_col1.metric(
                        "Performance Improvement",
                        f"{improvement:.1f}%"
                    )
                    
                    original_cost = optimization_details['original_analysis']['estimated_cost']
                    optimized_cost = optimization_details['optimized_analysis']['estimated_cost']
                    metric_col2.metric(
                        "Cost Savings",
                        f"${original_cost - optimized_cost:.4f}"
                    )
                    
                    metric_col3.metric(
                        "Optimized Cost",
                        f"${optimized_cost:.4f}"
                    )
                    
                    # Display optimized query
                    st.subheader("Optimized Query")
                    st.code(optimized_query, language="sql")
                    
                    # Show detailed analysis
                    with st.expander("View Detailed Analysis"):
                        st.markdown(optimization_details['optimized_analysis']['ai_analysis'])
                        
                except Exception as e:
                    st.error(f"Error during optimization: {str(e)}")
                    logger.error(f"Optimization error: {str(e)}")
        
        #Historical Performance
        st.header("Historical Performance")
        try:
            history_df = optimizer.get_performance_history()
            if not history_df.empty:
                st.plotly_chart(create_performance_chart(history_df))
            else:
                st.info("No historical data available yet.")
        except Exception as e:
            st.error(f"Error loading historical data: {str(e)}")
            logger.error(f"Historical data error: {str(e)}")

    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")

if __name__ == "__main__":
    main()