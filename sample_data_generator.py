# sample_data_generator.py - Generate Sample Data for Testing
"""
Generate sample data for testing the Operations Analyst Agent
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class SampleDataGenerator:
    """Generate realistic sample data for testing"""
    
    def __init__(self, start_date: str = "2024-01-01", num_customers: int = 50):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.num_customers = num_customers
        self.customers = self._generate_customers()
    
    def _generate_customers(self) -> pd.DataFrame:
        """Generate sample customer data"""
        customers = []
        tiers = ["enterprise", "pro", "standard", "basic"]
        
        for i in range(1, self.num_customers + 1):
            customer = {
                "id": f"C{i:03d}",
                "name": f"Customer {i}",
                "tier": random.choice(tiers),
                "signup_date": self.start_date + timedelta(days=random.randint(0, 365)),
                "status": random.choice(["active", "churned", "trial"]) if random.random() > 0.8 else "active"
            }
            customers.append(customer)
        
        return pd.DataFrame(customers)
    
    def generate_support_tickets(self, num_tickets: int = 500) -> pd.DataFrame:
        """Generate sample support ticket data"""
        tickets = []
        priorities = ["low", "medium", "high", "urgent"]
        categories = ["technical", "billing", "feature_request", "bug", "general"]
        statuses = ["open", "in_progress", "resolved", "closed"]
        
        for i in range(1, num_tickets + 1):
            created_date = self.start_date + timedelta(days=random.randint(0, 365))
            
            # Resolution time varies by priority
            priority = random.choice(priorities)
            if priority == "urgent":
                resolution_days = random.randint(0, 2)
            elif priority == "high":
                resolution_days = random.randint(1, 5)
            elif priority == "medium":
                resolution_days = random.randint(2, 10)
            else:
                resolution_days = random.randint(5, 15)
            
            status = random.choice(statuses)
            resolved_date = created_date + timedelta(days=resolution_days) if status in ["resolved", "closed"] else None
            
            ticket = {
                "id": f"T{i:04d}",
                "customer_id": random.choice(self.customers["id"]),
                "created_date": created_date.strftime("%Y-%m-%d"),
                "resolved_date": resolved_date.strftime("%Y-%m-%d") if resolved_date else None,
                "priority": priority,
                "category": random.choice(categories),
                "status": status,
                "satisfaction_score": random.randint(1, 5) if status in ["resolved", "closed"] else None
            }
            tickets.append(ticket)
        
        return pd.DataFrame(tickets)
    
    def generate_implementations(self, num_implementations: int = 100) -> pd.DataFrame:
        """Generate sample implementation data"""
        implementations = []
        statuses = ["not_started", "in_progress", "completed", "on_hold"]
        
        for i in range(1, num_implementations + 1):
            start_date = self.start_date + timedelta(days=random.randint(0, 300))
            
            # Implementation duration varies by customer tier
            customer_id = random.choice(self.customers["id"])
            customer_tier = self.customers[self.customers["id"] == customer_id]["tier"].iloc[0]
            
            if customer_tier == "enterprise":
                duration_days = random.randint(30, 90)
            elif customer_tier == "pro":
                duration_days = random.randint(20, 60)
            else:
                duration_days = random.randint(10, 30)
            
            status = random.choice(statuses)
            completion_pct = random.randint(0, 100) if status == "in_progress" else (100 if status == "completed" else 0)
            
            go_live_date = start_date + timedelta(days=duration_days) if status == "completed" else None
            
            implementation = {
                "id": f"I{i:03d}",
                "customer_id": customer_id,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "go_live_date": go_live_date.strftime("%Y-%m-%d") if go_live_date else None,
                "status": status,
                "completion_percentage": completion_pct
            }
            implementations.append(implementation)
        
        return pd.DataFrame(implementations)
    
    def generate_usage_metrics(self, num_records: int = 1000) -> pd.DataFrame:
        """Generate sample usage metrics data"""
        usage_records = []
        
        for i in range(1, num_records + 1):
            date = self.start_date + timedelta(days=random.randint(0, 365))
            customer_id = random.choice(self.customers["id"])
            
            # Usage varies by customer tier
            customer_tier = self.customers[self.customers["id"] == customer_id]["tier"].iloc[0]
            
            if customer_tier == "enterprise":
                base_users = random.randint(50, 200)
                base_sessions = random.randint(500, 2000)
            elif customer_tier == "pro":
                base_users = random.randint(20, 100)
                base_sessions = random.randint(200, 1000)
            else:
                base_users = random.randint(5, 50)
                base_sessions = random.randint(50, 500)
            
            usage = {
                "id": f"U{i:04d}",
                "customer_id": customer_id,
                "date": date.strftime("%Y-%m-%d"),
                "active_users": base_users + random.randint(-10, 10),
                "sessions": base_sessions + random.randint(-50, 50),
                "feature_usage_score": round(random.uniform(0.3, 1.0), 2)
            }
            usage_records.append(usage)
        
        return pd.DataFrame(usage_records)
    
    def save_all_sample_data(self, output_dir: str = "sample_data"):
        """Generate and save all sample data files"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Generate and save each dataset
        datasets = {
            "customers": self.customers,
            "support_tickets": self.generate_support_tickets(),
            "implementations": self.generate_implementations(),
            "usage_metrics": self.generate_usage_metrics()
        }
        
        for name, df in datasets.items():
            filepath = Path(output_dir) / f"{name}.csv"
            df.to_csv(filepath, index=False)
            print(f"Generated {len(df)} records for {name} -> {filepath}")
        
        print(f"\nSample data generated successfully in '{output_dir}' directory!")
        print("You can now upload these CSV files to test the Operations Analyst Agent.")

# data_schemas.py - Define data schemas and validation
"""
Data schemas and validation for the Operations Analyst Agent
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

class CustomerTier(str, Enum):
    ENTERPRISE = "enterprise"
    PRO = "pro"
    STANDARD = "standard"
    BASIC = "basic"

class CustomerStatus(str, Enum):
    ACTIVE = "active"
    CHURNED = "churned"
    TRIAL = "trial"

class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"

class Customer(BaseModel):
    id: str
    name: str
    tier: CustomerTier
    signup_date: datetime
    status: CustomerStatus

class SupportTicket(BaseModel):
    id: str
    customer_id: str
    created_date: datetime
    resolved_date: Optional[datetime] = None
    priority: TicketPriority
    category: str
    status: TicketStatus
    satisfaction_score: Optional[int] = Field(None, ge=1, le=5)

class Implementation(BaseModel):
    id: str
    customer_id: str
    start_date: datetime
    go_live_date: Optional[datetime] = None
    status: str
    completion_percentage: int = Field(ge=0, le=100)

class UsageMetric(BaseModel):
    id: str
    customer_id: str
    date: datetime
    active_users: int = Field(ge=0)
    sessions: int = Field(ge=0)
    feature_usage_score: float = Field(ge=0.0, le=1.0)

# Schema definitions for CSV validation
CSV_SCHEMAS = {
    "customers": {
        "required_columns": ["id", "name", "tier", "signup_date", "status"],
        "column_types": {
            "id": str,
            "name": str,
            "tier": str,
            "signup_date": "datetime",
            "status": str
        }
    },
    "support_tickets": {
        "required_columns": ["id", "customer_id", "created_date", "priority", "category", "status"],
        "column_types": {
            "id": str,
            "customer_id": str,
            "created_date": "datetime",
            "resolved_date": "datetime",
            "priority": str,
            "category": str,
            "status": str,
            "satisfaction_score": int
        }
    },
    "implementations": {
        "required_columns": ["id", "customer_id", "start_date", "status", "completion_percentage"],
        "column_types": {
            "id": str,
            "customer_id": str,
            "start_date": "datetime",
            "go_live_date": "datetime",
            "status": str,
            "completion_percentage": int
        }
    },
    "usage_metrics": {
        "required_columns": ["id", "customer_id", "date", "active_users", "sessions", "feature_usage_score"],
        "column_types": {
            "id": str,
            "customer_id": str,
            "date": "datetime",
            "active_users": int,
            "sessions": int,
            "feature_usage_score": float
        }
    }
}

# Usage example for generating sample data
if __name__ == "__main__":
    # Generate sample data
    generator = SampleDataGenerator(num_customers=25)
    generator.save_all_sample_data()
    
    print("\nTo use the sample data:")
    print("1. Run this script to generate sample CSV files")
    print("2. Launch the Streamlit app: streamlit run ops_agent_starter.py")
    print("3. Upload the generated CSV files through the sidebar")
    print("4. Explore the dashboards and insights!")
