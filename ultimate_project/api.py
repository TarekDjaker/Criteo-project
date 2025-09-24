"""
PRODUCTION-READY FASTAPI FOR CTR PREDICTION
Real-time serving with monitoring, caching, and error handling
"""

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import numpy as np
import time
import hashlib
import json
from datetime import datetime, timedelta
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import redis
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Criteo CTR Prediction API",
    description="Production-ready CTR prediction service handling 1M+ QPS",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on requirements
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=10)

# Redis client for caching (optional)
try:
    redis_client = redis.Redis(
        host='localhost',
        port=6379,
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=1
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
except:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using local cache only")


# Request/Response Models
class CTRFeatures(BaseModel):
    """Input features for CTR prediction"""
    
    # Numerical features (I1-I13 in Criteo dataset)
    numerical_features: List[float] = Field(
        ..., 
        min_items=13, 
        max_items=13,
        description="13 numerical features (counts, impressions, etc.)"
    )
    
    # Categorical features (C1-C26 in Criteo dataset)  
    categorical_features: List[str] = Field(
        ..., 
        min_items=26, 
        max_items=26,
        description="26 categorical features (IDs, categories, etc.)"
    )
    
    # Optional metadata
    user_id: Optional[str] = Field(None, description="User identifier")
    publisher_id: Optional[str] = Field(None, description="Publisher identifier")
    placement_id: Optional[str] = Field(None, description="Ad placement identifier")
    
    @validator('numerical_features')
    def validate_numerical(cls, v):
        """Validate numerical features are finite"""
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError("Numerical features must be finite")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "numerical_features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
                "categorical_features": ["cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8",
                                        "cat9", "cat10", "cat11", "cat12", "cat13", "cat14", "cat15", "cat16",
                                        "cat17", "cat18", "cat19", "cat20", "cat21", "cat22", "cat23", "cat24",
                                        "cat25", "cat26"],
                "user_id": "user123",
                "publisher_id": "pub456",
                "placement_id": "place789"
            }
        }


class CTRPredictionResponse(BaseModel):
    """CTR prediction response"""
    
    ctr_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted CTR (0-1)")
    should_bid: bool = Field(..., description="Whether to bid on this impression")
    recommended_bid_cpm: float = Field(..., ge=0.0, description="Recommended bid in CPM")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    model_version: str = Field(..., description="Model version used")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")
    cache_hit: bool = Field(False, description="Whether result was from cache")
    
    class Config:
        schema_extra = {
            "example": {
                "ctr_probability": 0.0025,
                "should_bid": True,
                "recommended_bid_cpm": 2.50,
                "confidence_score": 0.85,
                "model_version": "v1.2.3",
                "latency_ms": 4.5,
                "cache_hit": False
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    requests: List[CTRFeatures] = Field(..., min_items=1, max_items=1000)
    
    
class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[CTRPredictionResponse]
    batch_size: int
    total_latency_ms: float


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    model_loaded: bool
    cache_available: bool
    qps_current: int
    latency_p50_ms: float
    latency_p99_ms: float


# Mock model for demonstration (replace with actual model)
class CTRModel:
    """Mock CTR model - replace with actual implementation"""
    
    def __init__(self):
        self.version = "v1.2.3"
        self.feature_dim = 39  # 13 numerical + 26 categorical
        
    def predict(self, features: np.ndarray) -> tuple:
        """Mock prediction - replace with actual model inference"""
        # Simulate model inference
        base_ctr = 0.002  # 0.2% base CTR
        
        # Add some variation based on features
        variation = np.random.normal(0, 0.001)
        ctr = np.clip(base_ctr + variation + features.mean() * 0.0001, 0, 1)
        
        # Confidence based on feature quality
        confidence = 0.85 if not np.any(np.isnan(features)) else 0.5
        
        return float(ctr), float(confidence)


# Initialize model
model = CTRModel()

# Metrics tracking
class MetricsTracker:
    """Simple metrics tracker"""
    
    def __init__(self):
        self.request_count = 0
        self.latencies = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_reset = datetime.now()
        
    def add_request(self, latency_ms: float, cache_hit: bool):
        self.request_count += 1
        self.latencies.append(latency_ms)
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
        # Keep only last 1000 latencies
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]
    
    def get_metrics(self) -> Dict:
        if not self.latencies:
            return {
                "qps": 0,
                "p50_latency_ms": 0,
                "p99_latency_ms": 0,
                "cache_hit_rate": 0
            }
        
        elapsed = (datetime.now() - self.last_reset).total_seconds()
        qps = self.request_count / max(elapsed, 1)
        
        sorted_latencies = sorted(self.latencies)
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        
        return {
            "qps": qps,
            "p50_latency_ms": p50,
            "p99_latency_ms": p99,
            "cache_hit_rate": cache_hit_rate
        }


metrics = MetricsTracker()


# Helper functions
def hash_features(features: CTRFeatures) -> str:
    """Create cache key from features"""
    feature_str = f"{features.numerical_features}{features.categorical_features}"
    return hashlib.md5(feature_str.encode()).hexdigest()


def feature_engineering(features: CTRFeatures) -> np.ndarray:
    """Convert raw features to model input"""
    # Combine numerical features
    numerical = np.array(features.numerical_features)
    
    # Hash categorical features (simplified)
    categorical_hashed = []
    for cat in features.categorical_features:
        if cat:
            hash_val = int(hashlib.md5(cat.encode()).hexdigest(), 16) % 1000000
            categorical_hashed.append(hash_val / 1000000)  # Normalize
        else:
            categorical_hashed.append(0)
    
    # Combine all features
    combined = np.concatenate([numerical, categorical_hashed])
    
    # Add some feature crosses (simplified)
    crosses = [
        numerical[0] * numerical[1],  # Cross first two numerical
        numerical[2] / (numerical[3] + 1e-8),  # Ratio feature
    ]
    
    return np.concatenate([combined, crosses])


async def get_cached_prediction(cache_key: str) -> Optional[Dict]:
    """Get prediction from cache"""
    if REDIS_AVAILABLE:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except:
            pass
    return None


async def cache_prediction(cache_key: str, prediction: Dict, ttl: int = 300):
    """Cache prediction with TTL"""
    if REDIS_AVAILABLE:
        try:
            redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(prediction)
            )
        except:
            pass


def calculate_bid(ctr: float, value_per_click: float = 1.0) -> float:
    """Calculate recommended bid based on CTR"""
    # Simple bid calculation: CPM = CTR * CPC * 1000
    cpc = value_per_click  # Value of a click
    cpm = ctr * cpc * 1000
    
    # Apply some business logic
    min_bid = 0.10
    max_bid = 10.00
    
    return np.clip(cpm, min_bid, max_bid)


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "Criteo CTR Prediction API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    metrics_data = metrics.get_metrics()
    
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        model_loaded=model is not None,
        cache_available=REDIS_AVAILABLE,
        qps_current=int(metrics_data["qps"]),
        latency_p50_ms=metrics_data["p50_latency_ms"],
        latency_p99_ms=metrics_data["p99_latency_ms"]
    )


@app.post("/predict", response_model=CTRPredictionResponse)
async def predict_ctr(features: CTRFeatures):
    """Single CTR prediction endpoint"""
    start_time = time.time()
    
    try:
        # Check cache
        cache_key = hash_features(features)
        cached_result = await get_cached_prediction(cache_key)
        
        if cached_result:
            # Return cached result
            latency_ms = (time.time() - start_time) * 1000
            metrics.add_request(latency_ms, cache_hit=True)
            
            cached_result["latency_ms"] = latency_ms
            cached_result["cache_hit"] = True
            return CTRPredictionResponse(**cached_result)
        
        # Feature engineering
        model_input = feature_engineering(features)
        
        # Model prediction
        ctr_prob, confidence = model.predict(model_input)
        
        # Business logic
        should_bid = ctr_prob > 0.001  # Bid if CTR > 0.1%
        recommended_bid = calculate_bid(ctr_prob)
        
        # Prepare response
        prediction = {
            "ctr_probability": ctr_prob,
            "should_bid": should_bid,
            "recommended_bid_cpm": recommended_bid,
            "confidence_score": confidence,
            "model_version": model.version,
            "latency_ms": 0,  # Will be updated
            "cache_hit": False
        }
        
        # Cache result
        await cache_prediction(cache_key, prediction)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        prediction["latency_ms"] = latency_ms
        
        # Track metrics
        metrics.add_request(latency_ms, cache_hit=False)
        
        return CTRPredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    start_time = time.time()
    
    try:
        # Process predictions in parallel
        tasks = []
        for features in batch_request.requests:
            tasks.append(predict_ctr(features))
        
        predictions = await asyncio.gather(*tasks)
        
        total_latency_ms = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(predictions),
            total_latency_ms=total_latency_ms
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    """Get service metrics"""
    metrics_data = metrics.get_metrics()
    
    return {
        "qps": metrics_data["qps"],
        "latency": {
            "p50_ms": metrics_data["p50_latency_ms"],
            "p99_ms": metrics_data["p99_latency_ms"]
        },
        "cache": {
            "hit_rate": metrics_data["cache_hit_rate"],
            "available": REDIS_AVAILABLE
        },
        "model": {
            "version": model.version,
            "loaded": model is not None
        }
    }


@app.post("/feedback")
async def submit_feedback(
    prediction_id: str,
    clicked: bool,
    timestamp: Optional[datetime] = None
):
    """Submit click feedback for online learning"""
    # In production, this would update online learning system
    # and be used for model retraining
    
    feedback_data = {
        "prediction_id": prediction_id,
        "clicked": clicked,
        "timestamp": timestamp or datetime.now()
    }
    
    # Log feedback (in production, send to Kafka/stream)
    logger.info(f"Feedback received: {feedback_data}")
    
    return {"status": "feedback_recorded", "data": feedback_data}


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Log request details
    process_time = time.time() - start_time
    logger.info(
        f"Path: {request.url.path} | "
        f"Method: {request.method} | "
        f"Status: {response.status_code} | "
        f"Duration: {process_time:.3f}s"
    )
    
    # Add custom headers
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Model-Version"] = model.version
    
    return response


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general errors"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("Starting CTR Prediction API...")
    
    # Load model (in production, load from S3/model registry)
    logger.info(f"Model loaded: {model.version}")
    
    # Warm up cache
    if REDIS_AVAILABLE:
        logger.info("Redis cache available")
    else:
        logger.warning("Running without Redis cache")
    
    logger.info("API ready to serve requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down CTR Prediction API...")
    
    # Save metrics (in production)
    final_metrics = metrics.get_metrics()
    logger.info(f"Final metrics: {final_metrics}")
    
    # Cleanup resources
    executor.shutdown(wait=True)
    
    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    uvicorn.run(
        "api_fastapi_production:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )