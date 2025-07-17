# ops_agent_starter.py
"""
Operations Analyst Agent - Starter Implementation
This is a minimal viable implementation to get you started with CSV processing and basic analytics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
import os
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Handles CSV ingestion and basic data processing"""
    
    def __init__(self, db_path: str = "ops_agent.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        cursor = self.conn.cursor()
        
        # Create tables for different data types
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                id TEXT PRIMARY KEY,
                name TEXT,
                tier TEXT,
                signup_date DATE,
                status TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS support_tickets (
                id TEXT PRIMARY KEY,
                customer_id TEXT,
                created_date DATE,
                resolved_date DATE,
                priority TEXT,
                category TEXT,
                status TEXT,
                satisfaction_score INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS implementations (
                id TEXT PRIMARY KEY,
                customer_id TEXT,
                start_date DATE,
                go_live_date DATE,
                status TEXT,
                completion_percentage INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_metrics (
                id TEXT PRIMARY KEY,
                customer_id TEXT,
                date DATE,
                active_users INTEGER,
                sessions INTEGER,
                feature_usage_score REAL
            )
        ''')
        
        self.conn.commit()
    
    def process_csv(self, file_path: str, data_type: str) -> bool:
        """Process a CSV file and store in database"""
        try:
            df = pd.read_csv(file_path)
            
            # Basic data cleaning
            df = df.dropna(subset=[df.columns[0]])  # Remove rows with null primary key
            
            # Convert date columns
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            for col in date_columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Store in database
            df.to_sql(data_type, self.conn, if_exists='replace', index=False)
            
            print(f"Successfully processed {len(df)} rows from {file_path}")
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return False
    
    def get_data(self, table_name: str, limit: int = None) -> pd.DataFrame:
        """Retrieve data from database"""
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"
        return pd.read_sql_query(query, self.conn)

class MetricsCalculator:
    """Calculate key metrics and perform analysis"""
    
    def __init__(self, data_processor: DataProcessor):
        self.dp = data_processor
    
    def calculate_support_metrics(self) -> Dict:
        """Calculate support-related metrics"""
        try:
            tickets = self.dp.get_data('support_tickets')
            if tickets.empty:
                return {}
            
            tickets['created_date'] = pd.to_datetime(tickets['created_date'])
            tickets['resolved_date'] = pd.to_datetime(tickets['resolved_date'])
            
            # Calculate resolution time
            tickets['resolution_time'] = (tickets['resolved_date'] - tickets['created_date']).dt.days
            
            metrics = {
                'total_tickets': len(tickets),
                'avg_resolution_time': tickets['resolution_time'].mean(),
                'tickets_by_priority': tickets['priority'].value_counts().to_dict(),
                'avg_satisfaction': tickets['satisfaction_score'].mean(),
                'monthly_ticket_volume': tickets.groupby(tickets['created_date'].dt.to_period('M')).size().to_dict()
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating support metrics: {str(e)}")
            return {}
    
    def calculate_implementation_metrics(self) -> Dict:
        """Calculate implementation-related metrics"""
        try:
            implementations = self.dp.get_data('implementations')
            if implementations.empty:
                return {}
            
            implementations['start_date'] = pd.to_datetime(implementations['start_date'])
            implementations['go_live_date'] = pd.to_datetime(implementations['go_live_date'])
            
            # Calculate time to go-live
            implementations['time_to_golive'] = (implementations['go_live_date'] - implementations['start_date']).dt.days
            
            metrics = {
                'total_implementations': len(implementations),
                'avg_time_to_golive': implementations['time_to_golive'].mean(),
                'completion_rate': (implementations['status'] == 'completed').mean() * 100,
                'implementations_by_status': implementations['status'].value_counts().to_dict()
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating implementation metrics: {str(e)}")
            return {}
    
    def calculate_usage_metrics(self) -> Dict:
        """Calculate usage-related metrics"""
        try:
            usage = self.dp.get_data('usage_metrics')
            if usage.empty:
                return {}
            
            usage['date'] = pd.to_datetime(usage['date'])
            
            metrics = {
                'avg_active_users': usage['active_users'].mean(),
                'avg_sessions': usage['sessions'].mean(),
                'avg_feature_usage': usage['feature_usage_score'].mean(),
                'usage_trend': usage.groupby(usage['date'].dt.to_period('M'))['active_users'].mean().to_dict()
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating usage metrics: {str(e)}")
            return {}

class ForecastingEngine:
    """Simple forecasting using statistical methods"""
    
    def __init__(self, data_processor: DataProcessor):
        self.dp = data_processor
    
    def simple_forecast(self, data: pd.Series, periods: int = 6) -> Dict:
        """Simple moving average forecast"""
        try:
            # Calculate moving average
            window = min(12, len(data) // 2)  # Use 12 periods or half the data
            if window < 3:
                window = 3
            
            moving_avg = data.rolling(window=window).mean()
            
            # Calculate trend
            trend = (data.iloc[-3:].mean() - data.iloc[-6:-3].mean()) if len(data) >= 6 else 0
            
            # Generate forecast
            last_value = moving_avg.iloc[-1]
            forecast = []
            
            for i in range(periods):
                forecast_value = last_value + (trend * i)
                forecast.append(max(0, forecast_value))  # Ensure non-negative
            
            return {
                'historical': data.tolist(),
                'forecast': forecast,
                'confidence': 'medium' if len(data) > 12 else 'low'
            }
            
        except Exception as e:
            print(f"Error in forecasting: {str(e)}")
            return {'historical': [], 'forecast': [], 'confidence': 'low'}

class InsightsGenerator:
    """Generate insights from calculated metrics"""
    
    def __init__(self, metrics_calculator: MetricsCalculator):
        self.mc = metrics_calculator
    
    def generate_insights(self) -> List[str]:
        """Generate textual insights from metrics"""
        insights = []
        
        # Support insights
        support_metrics = self.mc.calculate_support_metrics()
        if support_metrics:
            avg_resolution = support_metrics.get('avg_resolution_time', 0)
            if avg_resolution > 3:
                insights.append(f"⚠️ Average resolution time is {avg_resolution:.1f} days - consider process optimization")
            
            satisfaction = support_metrics.get('avg_satisfaction', 0)
            if satisfaction < 4:
                insights.append(f"📊 Customer satisfaction is {satisfaction:.1f}/5 - focus on support quality")
        
        # Implementation insights
        impl_metrics = self.mc.calculate_implementation_metrics()
        if impl_metrics:
            completion_rate = impl_metrics.get('completion_rate', 0)
            if completion_rate < 80:
                insights.append(f"🚀 Implementation completion rate is {completion_rate:.1f}% - review onboarding process")
            
            time_to_golive = impl_metrics.get('avg_time_to_golive', 0)
            if time_to_golive > 60:
                insights.append(f"⏱️ Average time to go-live is {time_to_golive:.1f} days - streamline implementation")
        
        # Usage insights
        usage_metrics = self.mc.calculate_usage_metrics()
        if usage_metrics:
            feature_usage = usage_metrics.get('avg_feature_usage', 0)
            if feature_usage < 0.6:
                insights.append(f"📈 Feature usage score is {feature_usage:.1f} - enhance user training")
        
        if not insights:
            insights.append("✅ All metrics are within acceptable ranges")
        
        return insights

class Dashboard:
    """Create visualizations and dashboard"""
    
    def __init__(self, metrics_calculator: MetricsCalculator, forecasting_engine: ForecastingEngine):
        self.mc = metrics_calculator
        self.fe = forecasting_engine
    
    def create_support_dashboard(self):
        """Create support metrics dashboard"""
        metrics = self.mc.calculate_support_metrics()
        
        if not metrics:
            st.error("No support data available")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tickets", metrics.get('total_tickets', 0))
        
        with col2:
            st.metric("Avg Resolution Time", f"{metrics.get('avg_resolution_time', 0):.1f} days")
        
        with col3:
            st.metric("Avg Satisfaction", f"{metrics.get('avg_satisfaction', 0):.1f}/5")
        
        with col4:
            completion_rate = (metrics.get('total_tickets', 0) - len([t for t in metrics.get('tickets_by_priority', {}).values() if t == 'open'])) / max(metrics.get('total_tickets', 1), 1) * 100
            st.metric("Resolution Rate", f"{completion_rate:.1f}%")
        
        # Ticket volume over time
        monthly_volume = metrics.get('monthly_ticket_volume', {})
        if monthly_volume:
            df_volume = pd.DataFrame(list(monthly_volume.items()), columns=['Month', 'Tickets'])
            df_volume['Month'] = df_volume['Month'].astype(str)
            
            fig = px.line(df_volume, x='Month', y='Tickets', title='Monthly Ticket Volume')
            st.plotly_chart(fig, use_container_width=True)
    
    def create_implementation_dashboard(self):
        """Create implementation metrics dashboard"""
        metrics = self.mc.calculate_implementation_metrics()
        
        if not metrics:
            st.error("No implementation data available")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Implementations", metrics.get('total_implementations', 0))
        
        with col2:
            st.metric("Avg Time to Go-Live", f"{metrics.get('avg_time_to_golive', 0):.1f} days")
        
        with col3:
            st.metric("Completion Rate", f"{metrics.get('completion_rate', 0):.1f}%")
        
        # Status distribution
        status_dist = metrics.get('implementations_by_status', {})
        if status_dist:
            fig = px.pie(values=list(status_dist.values()), names=list(status_dist.keys()), 
                        title='Implementation Status Distribution')
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="Operations Analyst Agent", layout="wide")
    
    st.title("🤖 Operations Analyst Agent")
    st.markdown("Automated analysis and insights for Customer Success operations")
    
    # Initialize components
    data_processor = DataProcessor()
    metrics_calculator = MetricsCalculator(data_processor)
    forecasting_engine = ForecastingEngine(data_processor)
    insights_generator = InsightsGenerator(metrics_calculator)
    dashboard = Dashboard(metrics_calculator, forecasting_engine)
    
    # Sidebar for file uploads
    st.sidebar.title("📁 Data Upload")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV files", 
        type=['csv'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save uploaded file
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Determine data type based on filename
            data_type = "support_tickets"  # Default
            if "implementation" in uploaded_file.name.lower():
                data_type = "implementations"
            elif "usage" in uploaded_file.name.lower():
                data_type = "usage_metrics"
            elif "customer" in uploaded_file.name.lower():
                data_type = "customers"
            
            # Process the file
            if data_processor.process_csv(file_path, data_type):
                st.sidebar.success(f"✅ Processed {uploaded_file.name}")
            else:
                st.sidebar.error(f"❌ Failed to process {uploaded_file.name}")
            
            # Clean up temp file
            os.remove(file_path)
    
    # Main dashboard
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎫 Support", "🚀 Implementation", "💡 Insights"])
    
    with tab1:
        st.header("Key Performance Indicators")
        
        # Generate insights
        insights = insights_generator.generate_insights()
        st.subheader("🔍 Key Insights")
        for insight in insights:
            st.write(insight)
    
    with tab2:
        st.header("Support Metrics")
        dashboard.create_support_dashboard()
    
    with tab3:
        st.header("Implementation Metrics")
        dashboard.create_implementation_dashboard()
    
    with tab4:
        st.header("AI-Generated Insights")
        insights = insights_generator.generate_insights()
        
        for i, insight in enumerate(insights, 1):
            st.write(f"{i}. {insight}")
        
        # Simple forecast example
        st.subheader("📈 Sample Forecast")
        st.write("*Note: This is a simple statistical forecast. Advanced ML models will be added in future versions.*")
        
        # Create sample forecast visualization
        sample_data = pd.Series([10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38])
        forecast_result = forecasting_engine.simple_forecast(sample_data)
        
        if forecast_result['historical']:
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=list(range(len(forecast_result['historical']))),
                y=forecast_result['historical'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Forecast data
            forecast_x = list(range(len(forecast_result['historical']), 
                                  len(forecast_result['historical']) + len(forecast_result['forecast'])))
            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast_result['forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(title="Sample Metric Forecast", xaxis_title="Time Period", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
