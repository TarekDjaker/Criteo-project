"""
SYSTEM DESIGN: CTR PREDICTION AT 30B REQUESTS/DAY
Complete architecture for Criteo-scale operations
Interview-ready with diagrams and calculations
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math


@dataclass
class SystemRequirements:
    """Real Criteo scale requirements"""
    daily_requests = 30_000_000_000  # 30B/day
    qps_average = 350_000  # Average QPS
    qps_peak = 1_000_000  # Peak QPS
    latency_p50_ms = 5  # 50th percentile
    latency_p99_ms = 10  # 99th percentile
    availability = 0.9999  # 4 nines
    ctr_average = 0.002  # 0.2% CTR
    feature_size_bytes = 1000  # Per request
    model_size_gb = 10  # Trained model


class SystemArchitecture:
    """Complete system design for CTR at scale"""
    
    def __init__(self):
        self.requirements = SystemRequirements()
        
    def calculate_infrastructure(self) -> Dict:
        """Calculate infrastructure needs"""
        
        # 1. COMPUTE REQUIREMENTS
        # Assuming 1ms processing time per request
        servers_for_qps = math.ceil(self.requirements.qps_peak / 10000)  # 10K QPS per server
        
        # Add redundancy (N+2)
        total_servers = servers_for_qps + 2
        
        # 2. MEMORY REQUIREMENTS
        # Model in memory per server
        memory_per_server_gb = self.requirements.model_size_gb + 32  # 32GB for OS/buffer
        
        # Feature cache (1M requests * 1KB)
        cache_memory_gb = 1_000_000 * self.requirements.feature_size_bytes / (1024**3)
        
        # 3. STORAGE REQUIREMENTS
        # Daily logs
        daily_storage_tb = (self.requirements.daily_requests * 
                          self.requirements.feature_size_bytes / (1024**4))
        
        # With compression (10:1 ratio)
        compressed_storage_tb = daily_storage_tb / 10
        
        # 4. NETWORK REQUIREMENTS
        bandwidth_gbps = (self.requirements.qps_peak * 
                         self.requirements.feature_size_bytes * 8 / (1024**3))
        
        return {
            'compute': {
                'servers_needed': total_servers,
                'cores_per_server': 64,
                'total_cores': total_servers * 64
            },
            'memory': {
                'per_server_gb': memory_per_server_gb,
                'total_memory_tb': total_servers * memory_per_server_gb / 1024,
                'cache_size_gb': cache_memory_gb
            },
            'storage': {
                'daily_raw_tb': daily_storage_tb,
                'daily_compressed_tb': compressed_storage_tb,
                'yearly_compressed_pb': compressed_storage_tb * 365 / 1024
            },
            'network': {
                'bandwidth_gbps': bandwidth_gbps,
                'connections_per_second': self.requirements.qps_peak
            }
        }
    
    def get_architecture_components(self) -> str:
        """Return complete architecture design"""
        
        architecture = """
╔══════════════════════════════════════════════════════════════════╗
║                  CRITEO CTR SYSTEM ARCHITECTURE                   ║
║                    30B Requests/Day @ <10ms P99                   ║
╚══════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────┐
│                        TIER 1: EDGE LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   CDN    │  │   CDN    │  │   CDN    │  │   CDN    │       │
│  │  (US)    │  │  (EU)    │  │  (Asia)  │  │  (LATAM) │       │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘       │
│        │             │             │             │              │
│        └─────────────┴─────────────┴─────────────┘              │
│                             │                                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                     TIER 2: API GATEWAY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │            Load Balancer (L4/L7)                       │     │
│  │    - Geographic routing                                │     │
│  │    - SSL termination                                   │     │
│  │    - Rate limiting (per publisher)                    │     │
│  └────────────────┬───────────────────────────────────────┘     │
│                   │                                              │
│  ┌────────────────▼───────────────────────────────────────┐     │
│  │            API Gateway Cluster                         │     │
│  │    - Request validation                                │     │
│  │    - Feature extraction                                │     │
│  │    - Request batching                                  │     │
│  └────────────────┬───────────────────────────────────────┘     │
│                   │                                              │
└───────────────────┴──────────────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────────────┐
│                 TIER 3: PREDICTION SERVICE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Feature Store                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
│  │  │   Redis     │  │  Aerospike  │  │   DynamoDB  │      │   │
│  │  │  (Hot Data) │  │  (User Prof)│  │  (Historical)│     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼────────────────────────────────┐   │
│  │                Model Serving Layer                        │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐         │   │
│  │  │ TensorFlow │  │   PyTorch  │  │  LightGBM  │         │   │
│  │  │  Serving   │  │   Serve    │  │   Server   │         │   │
│  │  └────────────┘  └────────────┘  └────────────┘         │   │
│  │                                                           │   │
│  │  - Model A/B testing                                      │   │
│  │  - Online learning updates                                │   │
│  │  - Ensemble predictions                                   │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                             │                                    │
└─────────────────────────────┴────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────┐
│                    TIER 4: BIDDING ENGINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 Real-Time Bidder (RTB)                    │   │
│  │  - Bid price calculation: bid = CTR × CPC × 1000         │   │
│  │  - Budget pacing algorithms                               │   │
│  │  - Auction participation logic                            │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼────────────────────────────────┐   │
│  │              Campaign Management Service                   │   │
│  │  - Budget tracking                                        │   │
│  │  - Frequency capping                                      │   │
│  │  - Targeting rules                                        │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│                  TIER 5: DATA PIPELINE                           │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Stream Processing                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
│  │  │   Kafka     │→ │    Flink    │→ │  Cassandra  │      │   │
│  │  │  (Ingestion)│  │ (Processing)│  │  (Storage)  │      │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Batch Processing                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
│  │  │    HDFS     │→ │    Spark    │→ │    Hive     │      │   │
│  │  │   (Data)    │  │  (Training) │  │  (Warehouse)│      │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│                TIER 6: MONITORING & OPERATIONS                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Observability Stack                     │   │
│  │  • Metrics: Prometheus + Grafana                          │   │
│  │  • Logging: ELK Stack (Elasticsearch, Logstash, Kibana)   │   │
│  │  • Tracing: Jaeger / Zipkin                               │   │
│  │  • Alerting: PagerDuty + OpsGenie                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
"""
        return architecture
    
    def get_data_flow(self) -> str:
        """Data flow through the system"""
        
        flow = """
DATA FLOW SEQUENCE (Request to Response):

1. REQUEST INGESTION (0-1ms)
   Client → CDN → Load Balancer → API Gateway
   - Geographic routing
   - SSL termination
   - Request validation

2. FEATURE ENRICHMENT (1-2ms)
   API Gateway → Feature Store
   - User profile lookup (Redis)
   - Publisher data (Aerospike)
   - Historical CTR (DynamoDB)
   
3. PREDICTION (2-4ms)
   Feature Vector → Model Server
   - Ensemble of models
   - A/B test assignment
   - Score calculation
   
4. BIDDING DECISION (4-5ms)
   CTR Score → Bidding Engine
   - Budget check
   - Bid calculation
   - Auction participation
   
5. RESPONSE (5-6ms)
   Bid Response → Client
   - Async logging
   - Metrics emission

TOTAL LATENCY: <10ms P99

DATA PERSISTENCE FLOW:

1. REAL-TIME STREAM
   Events → Kafka → Flink → Feature Store
   - Click/impression events
   - User behavior updates
   - Campaign performance

2. BATCH PROCESSING
   HDFS → Spark → Model Training
   - Daily model retraining
   - Feature engineering
   - Performance analysis

3. ONLINE LEARNING
   Stream → Online Learner → Model Update
   - Incremental updates
   - Concept drift handling
   - Real-time adaptation
"""
        return flow
    
    def get_optimization_strategies(self) -> str:
        """Performance optimization techniques"""
        
        optimizations = """
PERFORMANCE OPTIMIZATIONS:

1. CACHING STRATEGIES
   • L1: Application cache (local memory) - 0.1ms
   • L2: Redis (hot data) - 1ms
   • L3: Aerospike (warm data) - 5ms
   • L4: DynamoDB (cold data) - 10ms
   
   Cache Hit Ratios:
   - User profiles: 95% (LRU with 1M entries)
   - Publisher data: 99% (static, refreshed hourly)
   - Model predictions: 80% (TTL: 5 minutes)

2. MODEL SERVING OPTIMIZATIONS
   • Quantization: INT8 inference (4x speedup)
   • Batching: Dynamic batching (10-100 requests)
   • GPU inference: For deep models only
   • Model pruning: 90% sparsity, 2x speedup
   • ONNX runtime: Cross-platform optimization

3. FEATURE ENGINEERING
   • Pre-computed features in Feature Store
   • Approximate algorithms (Count-Min Sketch)
   • Sampling for non-critical features
   • Vectorized operations (SIMD)
   • Feature hashing for categoricals

4. NETWORK OPTIMIZATIONS
   • Connection pooling
   • HTTP/2 with multiplexing
   • Protobuf for serialization
   • Regional deployments
   • Anycast routing

5. DATABASE OPTIMIZATIONS
   • Sharding by user_id (consistent hashing)
   • Read replicas for hot data
   • Materialized views for aggregates
   • Bloom filters for existence checks
   • Columnar storage for analytics

6. SCALING STRATEGIES
   • Horizontal scaling: Add servers
   • Vertical scaling: GPU for deep learning
   • Auto-scaling based on QPS
   • Spot instances for batch jobs
   • Reserved capacity for baseline

7. FAILURE HANDLING
   • Circuit breakers (Hystrix pattern)
   • Graceful degradation
   • Fallback to simple models
   • Request hedging for critical paths
   • Chaos engineering tests

COST OPTIMIZATIONS:

1. INFRASTRUCTURE
   • Spot instances: 70% cost reduction for batch
   • Reserved instances: 40% for baseline capacity
   • Auto-scaling: Scale down during low traffic
   • Multi-cloud: Leverage best prices

2. DATA STORAGE
   • Tiered storage (Hot/Warm/Cold)
   • Compression (10:1 for logs)
   • Data retention policies
   • Deduplication

3. MODEL SERVING
   • Model distillation (smaller models)
   • Edge inference where possible
   • Cached predictions
   • Approximate inference for low-value requests
"""
        return optimizations
    
    def get_monitoring_metrics(self) -> str:
        """Key metrics to monitor"""
        
        metrics = """
KEY MONITORING METRICS:

BUSINESS METRICS:
• CTR (Click-Through Rate)
  - Overall: Target > 0.2%
  - By segment: User, Publisher, Device
  - Trend: Hour-over-hour, Day-over-day
  
• Revenue Metrics
  - RPM (Revenue per 1000 impressions)
  - Fill rate
  - Win rate in auctions
  - Average bid price

• Campaign Performance
  - Budget utilization
  - Conversion rate
  - ROI/ROAS

TECHNICAL METRICS:
• Latency
  - P50: < 5ms
  - P95: < 8ms  
  - P99: < 10ms
  - P99.9: < 20ms
  
• Throughput
  - QPS: Current vs capacity
  - Request success rate: > 99.99%
  - Error rate by type
  
• Model Performance
  - AUC: > 0.80
  - LogLoss: < 0.45
  - Calibration ratio
  - Feature importance drift

• Infrastructure
  - CPU utilization: < 70%
  - Memory usage: < 80%
  - Network I/O
  - Disk I/O
  - Cache hit rates

• Data Pipeline
  - Kafka lag
  - Processing delay
  - Data completeness
  - Schema violations

ALERTING THRESHOLDS:
• Critical: CTR drop > 20%, Latency P99 > 15ms
• High: Error rate > 1%, CPU > 90%
• Medium: Cache hit < 80%, Model drift detected
• Low: Disk usage > 70%, Training delay > 1hr

DASHBOARDS:
1. Executive Dashboard
   - Revenue, CTR, Fill rate
   - Top campaigns performance

2. Operations Dashboard
   - System health, Latency, QPS
   - Error rates, Alerts

3. Model Dashboard
   - Model versions, A/B tests
   - Performance metrics, Drift

4. Publisher Dashboard
   - Publisher-specific metrics
   - Revenue, CTR by placement
"""
        return metrics


def generate_capacity_planning():
    """Generate capacity planning calculations"""
    
    arch = SystemArchitecture()
    infra = arch.calculate_infrastructure()
    
    planning = f"""
╔══════════════════════════════════════════════════════════════════╗
║                     CAPACITY PLANNING                             ║
╚══════════════════════════════════════════════════════════════════╝

TRAFFIC ANALYSIS:
• Daily Requests: {arch.requirements.daily_requests:,}
• Average QPS: {arch.requirements.qps_average:,}
• Peak QPS: {arch.requirements.qps_peak:,}
• Peak/Average Ratio: {arch.requirements.qps_peak/arch.requirements.qps_average:.1f}x

COMPUTE REQUIREMENTS:
• Servers Needed: {infra['compute']['servers_needed']}
• Cores per Server: {infra['compute']['cores_per_server']}
• Total Cores: {infra['compute']['total_cores']:,}
• Redundancy: N+2

MEMORY REQUIREMENTS:
• Per Server: {infra['memory']['per_server_gb']} GB
• Total Memory: {infra['memory']['total_memory_tb']:.1f} TB
• Cache Size: {infra['memory']['cache_size_gb']:.1f} GB

STORAGE REQUIREMENTS:
• Daily Raw Data: {infra['storage']['daily_raw_tb']:.1f} TB
• Daily Compressed: {infra['storage']['daily_compressed_tb']:.1f} TB
• Yearly Storage: {infra['storage']['yearly_compressed_pb']:.1f} PB

NETWORK REQUIREMENTS:
• Bandwidth: {infra['network']['bandwidth_gbps']:.1f} Gbps
• Connections/sec: {infra['network']['connections_per_second']:,}

COST ESTIMATION (AWS):
• Compute (EC2): ~${infra['compute']['servers_needed'] * 5000}/month
• Storage (S3): ~${infra['storage']['daily_compressed_tb'] * 30 * 23}/month
• Network (Transfer): ~${infra['network']['bandwidth_gbps'] * 100}/month
• Total: ~${infra['compute']['servers_needed'] * 5000 + infra['storage']['daily_compressed_tb'] * 30 * 23 + infra['network']['bandwidth_gbps'] * 100:,}/month
"""
    return planning


def interview_talking_points():
    """Key points to discuss in interview"""
    
    points = """
╔══════════════════════════════════════════════════════════════════╗
║                    INTERVIEW TALKING POINTS                       ║
╚══════════════════════════════════════════════════════════════════╝

1. SCALABILITY CHALLENGES
   • "At 30B requests/day, we need 1M QPS peak capacity"
   • "Horizontal scaling with consistent hashing for sharding"
   • "Multi-region deployment for global latency < 50ms"

2. LATENCY OPTIMIZATIONS  
   • "Every millisecond counts - 100ms delay = 1% revenue loss"
   • "Feature caching saves 3-4ms per request"
   • "Model quantization gives 4x speedup with minimal accuracy loss"

3. REAL-TIME REQUIREMENTS
   • "Sub-10ms response time for 100ms RTB timeout"
   • "Online learning for immediate feedback incorporation"
   • "Stream processing for real-time feature updates"

4. COST EFFICIENCY
   • "Spot instances for 70% cost reduction on batch jobs"
   • "Tiered storage: hot (Redis), warm (Aerospike), cold (S3)"
   • "Model distillation reduces serving costs by 50%"

5. RELIABILITY
   • "99.99% availability = 52 minutes downtime/year max"
   • "Multi-region failover with <1 minute RTO"
   • "Graceful degradation - fallback to simple models"

6. MACHINE LEARNING AT SCALE
   • "Distributed training on Spark for TB-scale data"
   • "A/B testing framework for continuous improvement"
   • "Feature store for consistency between training/serving"

7. PRIVACY & COMPLIANCE
   • "GDPR compliance with user consent management"
   • "Differential privacy for user features"
   • "Data retention policies and right-to-be-forgotten"

8. INNOVATION OPPORTUNITIES
   • "Transformer models for sequential user behavior"
   • "Federated learning for privacy-preserving training"
   • "Reinforcement learning for bid optimization"

CRITEO-SPECIFIC INSIGHTS:
• "Criteo processes 30B+ requests daily across 20K+ publishers"
• "Sub-2ms model inference using optimized C++ implementations"
• "Proprietary Criteo Engine handles 600K QPS per datacenter"
• "DeepKNN for candidate retrieval from 1B+ products"
"""
    return points


if __name__ == "__main__":
    print("\n🏗️ SYSTEM DESIGN: CTR @ CRITEO SCALE\n")
    
    arch = SystemArchitecture()
    
    # Show architecture
    print(arch.get_architecture_components())
    
    # Show data flow
    print("\n" + "=" * 70)
    print(arch.get_data_flow())
    
    # Show optimizations
    print("\n" + "=" * 70)
    print(arch.get_optimization_strategies())
    
    # Show monitoring
    print("\n" + "=" * 70)
    print(arch.get_monitoring_metrics())
    
    # Show capacity planning
    print("\n" + "=" * 70)
    print(generate_capacity_planning())
    
    # Show interview points
    print("\n" + "=" * 70)
    print(interview_talking_points())
    
    print("\n" + "=" * 70)
    print("✅ SYSTEM DESIGN MODULE COMPLETE!")
    print("=" * 70)
    print("\n🎯 Ready for Criteo system design interview!")
    print("  • Complete architecture documented")
    print("  • Scalability calculations ready")
    print("  • Cost optimizations identified")
    print("  • Monitoring strategy defined")
    print("  • Interview talking points prepared")