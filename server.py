from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class OutlierDetectionRequest(BaseModel):
    file_id: str
    threshold_percentile: float = 95.0

class OutlierDetectionResponse(BaseModel):
    file_id: str
    total_records: int
    outliers_count: int
    cleaned_records: int
    threshold: float
    visualizations: Dict[str, str]  # base64 encoded images
    reconstruction_errors: List[float]

# In-memory storage for uploaded files and results
file_storage = {}
results_storage = {}

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Scavenger API - Outlier Detection Service"}

@api_router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV file for outlier detection"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read file content
        contents = await file.read()
        
        # Parse CSV
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")
        
        # Validate data
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Generate file ID
        file_id = str(uuid.uuid4())
        
        # Store file data
        file_storage[file_id] = {
            'filename': file.filename,
            'data': df,
            'uploaded_at': datetime.now(timezone.utc).isoformat()
        }
        
        return {
            'file_id': file_id,
            'filename': file.filename,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@api_router.post("/detect-outliers", response_model=OutlierDetectionResponse)
async def detect_outliers(request: OutlierDetectionRequest):
    """Run autoencoder-based outlier detection"""
    try:
        # Get file data
        if request.file_id not in file_storage:
            raise HTTPException(status_code=404, detail="File not found")
        
        df = file_storage[request.file_id]['data'].copy()
        
        # Drop any NaN values
        df = df.dropna()
        
        # Prepare data - scale all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise HTTPException(status_code=400, detail="No numeric columns found in the dataset")
        
        df_numeric = df[numeric_cols]
        
        # Scale data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_numeric)
        
        # Split data
        X_train, X_test = train_test_split(df_scaled, test_size=0.2, random_state=42)
        
        # Build autoencoder
        input_dim = X_train.shape[1]
        autoencoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='linear')
        ])
        
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train autoencoder
        autoencoder.fit(X_train, X_train, epochs=25, batch_size=256, verbose=0)
        
        # Predict and calculate reconstruction error
        X_pred = autoencoder.predict(X_test, verbose=0)
        reconstruction_error = np.mean(np.square(X_test - X_pred), axis=1)
        
        # Determine threshold
        threshold = np.percentile(reconstruction_error, request.threshold_percentile)
        
        # Identify outliers
        outliers = reconstruction_error > threshold
        cleaned_data = X_test[~outliers]
        
        # Generate visualizations
        visualizations = {}
        
        # 1. Scatter plot (first two features)
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(cleaned_data[:, 0], cleaned_data[:, 1], c='#4FB3BF', 
                   label='Cleaned Data', alpha=0.6, s=30)
        ax1.scatter(X_test[outliers, 0], X_test[outliers, 1], c='#FF6B6B', 
                   label='Outliers', alpha=0.8, s=50, marker='x')
        ax1.set_xlabel('Feature 1 (Scaled)', fontsize=12)
        ax1.set_ylabel('Feature 2 (Scaled)', fontsize=12)
        ax1.set_title('Outlier Detection Results', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
        buf1.seek(0)
        visualizations['scatter_plot'] = base64.b64encode(buf1.read()).decode('utf-8')
        plt.close(fig1)
        
        # 2. Reconstruction error distribution
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.hist(reconstruction_error, bins=50, color='#4FB3BF', alpha=0.7, edgecolor='black')
        ax2.axvline(threshold, color='#FF6B6B', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
        ax2.set_xlabel('Reconstruction Error', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
        buf2.seek(0)
        visualizations['error_distribution'] = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close(fig2)
        
        # 3. Feature-wise comparison (first 4 features)
        num_features = min(4, X_test.shape[1])
        fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i in range(num_features):
            axes[i].hist(X_test[~outliers, i], bins=30, color='#4FB3BF', alpha=0.7, label='Cleaned Data')
            axes[i].hist(X_test[outliers, i], bins=30, color='#FF6B6B', alpha=0.7, label='Outliers')
            axes[i].set_xlabel(f'Feature {i+1} Value', fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
            axes[i].set_title(f'Feature {i+1} Distribution', fontsize=11, fontweight='bold')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for i in range(num_features, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png', dpi=100, bbox_inches='tight')
        buf3.seek(0)
        visualizations['feature_distribution'] = base64.b64encode(buf3.read()).decode('utf-8')
        plt.close(fig3)
        
        # Store results
        results_storage[request.file_id] = {
            'outlier_indices': np.where(outliers)[0].tolist(),
            'cleaned_data': cleaned_data,
            'scaler': scaler,
            'threshold': threshold
        }
        
        return OutlierDetectionResponse(
            file_id=request.file_id,
            total_records=len(X_test),
            outliers_count=int(np.sum(outliers)),
            cleaned_records=len(cleaned_data),
            threshold=float(threshold),
            visualizations=visualizations,
            reconstruction_errors=reconstruction_error.tolist()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in outlier detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error detecting outliers: {str(e)}")

@api_router.get("/download-cleaned/{file_id}")
async def download_cleaned(file_id: str):
    """Download cleaned dataset without outliers"""
    try:
        if file_id not in file_storage:
            raise HTTPException(status_code=404, detail="File not found")
        
        if file_id not in results_storage:
            raise HTTPException(status_code=400, detail="Please run outlier detection first")
        
        # Get original data
        df = file_storage[file_id]['data'].copy()
        df = df.dropna()
        
        # Get outlier indices from test set
        outlier_indices = results_storage[file_id]['outlier_indices']
        
        # Since we used train_test_split with test_size=0.2, we need to recreate the split
        # to get the correct indices
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df[numeric_cols]
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_numeric)
        _, X_test_indices = train_test_split(range(len(df_scaled)), test_size=0.2, random_state=42)
        
        # Map outlier indices to original dataframe indices
        actual_outlier_indices = [X_test_indices[i] for i in outlier_indices]
        
        # Remove outliers
        df_cleaned = df.drop(df.index[actual_outlier_indices])
        
        # Convert to CSV
        output = io.StringIO()
        df_cleaned.to_csv(output, index=False)
        output.seek(0)
        
        # Create response
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=cleaned_{file_storage[file_id]['filename']}"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error downloading cleaned data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading cleaned data: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()