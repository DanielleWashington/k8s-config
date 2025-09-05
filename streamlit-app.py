"""
Weaviate Sizing Tool
A tool for estimating resource requirements and generating a configuration for your Weaviate deployment.
Inspired by the Milvus sizing tool interface and functionality.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import yaml
from datetime import datetime
import zipfile
import io
import math

# ===== CONFIGURATION AND CONSTANTS =====
class IndexType(Enum):
    HNSW = "hnsw"
    FLAT = "flat"
    
class CompressionType(Enum):
    NONE = "none"
    PQ = "pq"  # Product Quantization
    BQ = "bq"  # Binary Quantization
    SQ = "sq"  # Scalar Quantization

class DeploymentType(Enum):
    STANDALONE = "standalone"
    DISTRIBUTED = "distributed"

@dataclass
class ResourceEstimate:
    """Comprehensive resource estimation results"""
    # Data sizes
    raw_data_size_gb: float
    loading_memory_gb: float
    
    # Resource requirements
    cpu_cores: float
    memory_gb: float
    storage_gb: float
    local_disk_gb: float
    
    # Compressed variants
    memory_gb_compressed: float
    
    # Breakdown components
    vector_memory_gb: float
    index_memory_gb: float
    metadata_memory_gb: float
    
    # Performance metrics
    estimated_qps: float
    estimated_latency_ms: float

# Embedding models with realistic dimensions
EMBEDDING_MODELS = {
    "OpenAI": {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536
    },
    "Cohere": {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384
    },
    "Google": {
        "textembedding-gecko": 768,
        "textembedding-gecko-multilingual": 768
    },
    "Sentence Transformers": {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "all-MiniLM-L12-v2": 384
    }
}

# Index type information
INDEX_INFO = {
    IndexType.HNSW: {
        "name": "HNSW",
        "description": "Hierarchical Navigable Small World - Balanced performance and memory usage",
        "memory_multiplier": 1.5,  # Additional memory for graph structure
        "query_speed": "Very Fast",
        "accuracy": "High",
        "build_time": "Medium",
        "recommended": True
    },
    IndexType.FLAT: {
        "name": "FLAT",
        "description": "Brute-force exact search - Highest accuracy, slowest queries",
        "memory_multiplier": 1.0,  # Just the vectors
        "query_speed": "Slow",
        "accuracy": "Perfect",
        "build_time": "None",
        "recommended": False
    }
}

COMPRESSION_INFO = {
    CompressionType.NONE: {
        "name": "No Compression",
        "memory_reduction": 0,
        "accuracy_impact": "None",
        "setup_complexity": "None"
    },
    CompressionType.PQ: {
        "name": "Product Quantization (PQ)",
        "memory_reduction": 85,
        "accuracy_impact": "Minimal (1-3%)",
        "setup_complexity": "Medium"
    },
    CompressionType.BQ: {
        "name": "Binary Quantization (BQ)", 
        "memory_reduction": 97,
        "accuracy_impact": "Low (5-10%)",
        "setup_complexity": "Low"
    },
    CompressionType.SQ: {
        "name": "Scalar Quantization (SQ)",
        "memory_reduction": 75,
        "accuracy_impact": "Very Low (1-2%)",
        "setup_complexity": "Low"
    }
}

class WeaviateResourceCalculator:
    """Advanced Weaviate resource calculator with Milvus-style functionality"""
    
    def __init__(self):
        self.BYTES_PER_FLOAT32 = 4
        self.GC_MULTIPLIER = 2.0  # Go garbage collection overhead
        self.DISK_OVERHEAD = 1.2
        
    def calculate_raw_data_size(self, num_vectors: int, dimensions: int, 
                               avg_metadata_bytes: int = 1024) -> float:
        """Calculate raw data size in GB"""
        vector_bytes = num_vectors * dimensions * self.BYTES_PER_FLOAT32
        metadata_bytes = num_vectors * avg_metadata_bytes
        return (vector_bytes + metadata_bytes) / (1024**3)
    
    def calculate_loading_memory(self, raw_data_gb: float, index_type: IndexType,
                                compression: CompressionType) -> float:
        """Calculate memory needed during loading"""
        index_multiplier = INDEX_INFO[index_type]["memory_multiplier"]
        base_memory = raw_data_gb * index_multiplier * self.GC_MULTIPLIER
        
        # Apply compression
        if compression != CompressionType.NONE:
            reduction = COMPRESSION_INFO[compression]["memory_reduction"] / 100
            base_memory = base_memory * (1 - reduction)
            
        return base_memory
    
    def calculate_cpu_requirements(self, num_vectors: int, target_qps: int = 50) -> Tuple[float, float]:
        """Calculate CPU requirements and estimate performance"""
        # Base CPU calculation (Weaviate benchmarks)
        base_cores_per_million = 2.0  # cores per 1M vectors
        total_cores = (num_vectors / 1_000_000) * base_cores_per_million
        total_cores = max(2, total_cores)  # Minimum 2 cores
        
        # Estimate QPS and latency
        estimated_qps_per_core = 25  # Conservative estimate
        max_qps = total_cores * estimated_qps_per_core
        estimated_latency = 1000 / estimated_qps_per_core  # ms
        
        return total_cores, max_qps, estimated_latency
    
    def calculate_storage_requirements(self, raw_data_gb: float, 
                                     enable_disk_offload: bool = False) -> Tuple[float, float]:
        """Calculate storage and local disk requirements"""
        storage_gb = raw_data_gb * self.DISK_OVERHEAD
        
        if enable_disk_offload:
            # With disk offloading, some data can be kept on slower storage
            local_disk_gb = raw_data_gb * 0.3  # 30% hot data
            storage_gb = raw_data_gb * 0.7 * self.DISK_OVERHEAD  # 70% cold data
        else:
            local_disk_gb = storage_gb
            
        return storage_gb, local_disk_gb
    
    def get_resource_estimate(self, num_vectors: int, dimensions: int,
                             index_type: IndexType = IndexType.HNSW,
                             compression: CompressionType = CompressionType.NONE,
                             avg_metadata_bytes: int = 1024,
                             enable_disk_offload: bool = False,
                             target_qps: int = 50) -> ResourceEstimate:
        """Generate comprehensive resource estimate"""
        
        # Calculate base data size
        raw_data_gb = self.calculate_raw_data_size(num_vectors, dimensions, avg_metadata_bytes)
        
        # Calculate memory requirements
        loading_memory_gb = self.calculate_loading_memory(raw_data_gb, index_type, compression)
        
        # Memory breakdown
        vector_memory_gb = (num_vectors * dimensions * self.BYTES_PER_FLOAT32) / (1024**3)
        index_memory_gb = vector_memory_gb * (INDEX_INFO[index_type]["memory_multiplier"] - 1)
        metadata_memory_gb = (num_vectors * avg_metadata_bytes) / (1024**3)
        
        # Apply compression to vector memory
        if compression != CompressionType.NONE:
            reduction = COMPRESSION_INFO[compression]["memory_reduction"] / 100
            vector_memory_compressed = vector_memory_gb * (1 - reduction)
            memory_gb_compressed = (vector_memory_compressed * self.GC_MULTIPLIER) + index_memory_gb + metadata_memory_gb
        else:
            memory_gb_compressed = loading_memory_gb
        
        # Calculate CPU and performance
        cpu_cores, estimated_qps, estimated_latency = self.calculate_cpu_requirements(num_vectors, target_qps)
        
        # Calculate storage
        storage_gb, local_disk_gb = self.calculate_storage_requirements(raw_data_gb, enable_disk_offload)
        
        return ResourceEstimate(
            raw_data_size_gb=raw_data_gb,
            loading_memory_gb=loading_memory_gb,
            cpu_cores=cpu_cores,
            memory_gb=loading_memory_gb,
            storage_gb=storage_gb,
            local_disk_gb=local_disk_gb,
            memory_gb_compressed=memory_gb_compressed,
            vector_memory_gb=vector_memory_gb,
            index_memory_gb=index_memory_gb,
            metadata_memory_gb=metadata_memory_gb,
            estimated_qps=estimated_qps,
            estimated_latency_ms=estimated_latency
        )

def generate_docker_compose(config: dict) -> str:
    """Generate Docker Compose configuration for Weaviate"""
    memory_limit = f"{int(config['memory_gb'])}g"
    cpu_limit = config['cpu_cores']
    
    return f"""version: '3.8'
services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-openai'
      ENABLE_MODULES: 'text2vec-openai,text2vec-cohere,text2vec-huggingface'
      CLUSTER_HOSTNAME: 'node1'
      LIMIT_RESOURCES: 'true'
      GOMEMLIMIT: '{memory_limit}'
    volumes:
      - weaviate_data:/var/lib/weaviate
    deploy:
      resources:
        limits:
          memory: {memory_limit}
          cpus: '{cpu_limit}'
        reservations:
          memory: {int(config['memory_gb'] * 0.8)}g
          cpus: '{cpu_limit * 0.5}'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/v1/.well-known/ready"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 40s

volumes:
  weaviate_data:

# Generated for {config['num_vectors']:,} vectors with {config['dimensions']} dimensions
# Estimated memory usage: {config['memory_gb']:.1f} GB
# Estimated CPU usage: {config['cpu_cores']:.1f} cores
# Index type: {config['index_type']}
# Compression: {config['compression']}
"""

def generate_kubernetes_manifest(config: dict) -> str:
    """Generate Kubernetes manifest for Weaviate"""
    memory_request = f"{int(config['memory_gb'] * 0.8)}Gi"
    memory_limit = f"{int(config['memory_gb'])}Gi"
    cpu_request = f"{int(config['cpu_cores'] * 0.5)}"
    cpu_limit = f"{int(config['cpu_cores'])}"
    storage_size = f"{int(config['storage_gb'])}Gi"
    
    return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: weaviate
  labels:
    app: weaviate
    config-hash: "{hash(str(config)) % 10000:04d}"
spec:
  replicas: 1
  selector:
    matchLabels:
      app: weaviate
  template:
    metadata:
      labels:
        app: weaviate
    spec:
      containers:
      - name: weaviate
        image: semitechnologies/weaviate:latest
        ports:
        - containerPort: 8080
          name: http
        resources:
          requests:
            memory: "{memory_request}"
            cpu: "{cpu_request}"
          limits:
            memory: "{memory_limit}"
            cpu: "{cpu_limit}"
        env:
        - name: QUERY_DEFAULTS_LIMIT
          value: "25"
        - name: AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED
          value: "true"
        - name: PERSISTENCE_DATA_PATH
          value: "/var/lib/weaviate"
        - name: DEFAULT_VECTORIZER_MODULE
          value: "text2vec-openai"
        - name: ENABLE_MODULES
          value: "text2vec-openai,text2vec-cohere,text2vec-huggingface"
        - name: CLUSTER_HOSTNAME
          value: "node1"
        - name: LIMIT_RESOURCES
          value: "true"
        - name: GOMEMLIMIT
          value: "{memory_limit.replace('Gi', 'GB')}"
        volumeMounts:
        - name: weaviate-data
          mountPath: /var/lib/weaviate
        livenessProbe:
          httpGet:
            path: /v1/.well-known/live
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /v1/.well-known/ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
      volumes:
      - name: weaviate-data
        persistentVolumeClaim:
          claimName: weaviate-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: weaviate-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: {storage_size}
  storageClassName: gp3
---
apiVersion: v1
kind: Service
metadata:
  name: weaviate
  labels:
    app: weaviate
spec:
  ports:
  - port: 8080
    targetPort: 8080
    name: http
  selector:
    app: weaviate
  type: ClusterIP

# Configuration Summary:
# Vectors: {config['num_vectors']:,}
# Dimensions: {config['dimensions']}
# Index: {config['index_type']}
# Compression: {config['compression']}
# Memory: {config['memory_gb']:.1f} GB
# CPU: {config['cpu_cores']:.1f} cores
# Storage: {config['storage_gb']:.1f} GB
"""

def format_bytes(bytes_val: float) -> str:
    """Format bytes with appropriate units"""
    if bytes_val == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"

def format_number(num: float) -> str:
    """Format numbers with appropriate suffixes"""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:.0f}"

def main():
    # Page configuration - clean and professional
    st.set_page_config(
        page_title="Weaviate Sizing Tool",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for Milvus-like styling
    st.markdown("""
    <style>
        /* Main styling */
        .main-container {
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header-section {
            text-align: center;
            margin-bottom: 2rem;
            padding: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
        }
        .section-header {
            font-size: 1.2rem;
            font-weight: 600;
            margin: 1.5rem 0 1rem 0;
            color: #2c3e50;
            border-left: 4px solid #667eea;
            padding-left: 1rem;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid #e9ecef;
            text-align: center;
            margin-bottom: 1rem;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #495057;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #6c757d;
            margin-top: 0.5rem;
        }
        .info-box {
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
        }
        .warning-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
        }
        .stSlider > div > div > div > div {
            background-color: #667eea;
        }
        .download-section {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            margin-top: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="header-section">
        <h1>🔍 Weaviate Sizing Tool</h1>
        <p>A tool for estimating resource requirements and generating a configuration for your Weaviate deployment.</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize calculator
    calculator = WeaviateResourceCalculator()
    
    # Main input section
    st.markdown('<div class="section-header">📊 Configuration Parameters</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Number of Vectors**")
        # Vector count slider (Milvus-style)
        vector_options = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
        vector_labels = ["10K", "100K", "1M", "10M", "100M", "1B"]
        
        vector_index = st.select_slider(
            "",
            options=list(range(len(vector_options))),
            value=2,  # Default to 1M
            format_func=lambda x: vector_labels[x],
            key="vector_count"
        )
        num_vectors = vector_options[vector_index]
        st.info(f"Selected: **{format_number(num_vectors)} vectors**")
        
        st.markdown("**Vector Dimensions**")
        # Dimension options
        dimension_preset = st.selectbox(
            "Choose embedding model or custom:",
            ["Custom"] + [f"{provider} - {model}" for provider in EMBEDDING_MODELS 
                         for model in EMBEDDING_MODELS[provider]]
        )
        
        if dimension_preset == "Custom":
            dimensions = st.slider("Custom Dimensions", 128, 4096, 768, step=64)
        else:
            provider, model = dimension_preset.split(" - ", 1)
            dimensions = EMBEDDING_MODELS[provider][model]
            st.info(f"**{dimensions} dimensions** for {model}")

    with col2:
        st.markdown("**Index Type**")
        index_type = st.radio(
            "",
            [IndexType.HNSW, IndexType.FLAT],
            format_func=lambda x: INDEX_INFO[x]["name"],
            help="HNSW recommended for most use cases"
        )
        
        # Show index info
        info = INDEX_INFO[index_type]
        st.markdown(f"""
        **{info['name']}**: {info['description']}
        - **Query Speed**: {info['query_speed']}
        - **Accuracy**: {info['accuracy']}
        - **Build Time**: {info['build_time']}
        """)
        
        st.markdown("**Vector Compression**")
        compression = st.selectbox(
            "",
            list(CompressionType),
            format_func=lambda x: COMPRESSION_INFO[x]["name"]
        )
        
        if compression != CompressionType.NONE:
            comp_info = COMPRESSION_INFO[compression]
            st.success(f"**{comp_info['memory_reduction']}%** memory reduction")

    # Advanced options
    with st.expander("⚙️ Advanced Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Scalar Fields**")
            has_scalar_fields = st.checkbox("Include scalar/metadata fields")
            avg_metadata_bytes = 1024  # Default
            if has_scalar_fields:
                avg_metadata_bytes = st.slider(
                    "Average metadata size per vector (bytes)",
                    100, 10000, 1024, step=100
                )
        
        with col2:
            st.markdown("**Deployment Options**")
            enable_disk_offload = st.checkbox(
                "Enable disk offloading",
                help="Use memory mapping for large datasets to reduce RAM usage"
            )
            
            deployment_type = st.radio(
                "Deployment Architecture:",
                [DeploymentType.STANDALONE, DeploymentType.DISTRIBUTED],
                format_func=lambda x: x.value.title(),
                horizontal=True
            )

    # Calculate results
    st.markdown('<div class="section-header">📈 Resource Requirements</div>', unsafe_allow_html=True)
    
    # Perform calculation
    results = calculator.get_resource_estimate(
        num_vectors=num_vectors,
        dimensions=dimensions,
        index_type=index_type,
        compression=compression,
        avg_metadata_bytes=avg_metadata_bytes,
        enable_disk_offload=enable_disk_offload
    )
    
    # Display results in Milvus-style layout
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        st.markdown("**Data Size Calculation**")
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{format_bytes(results.raw_data_size_gb * 1024**3)}</div>
            <div class="metric-label">Raw Data Size</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{format_bytes(results.loading_memory_gb * 1024**3)}</div>
            <div class="metric-label">Loading Memory</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("**Compute Resources**")
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{results.cpu_cores:.1f}</div>
            <div class="metric-label">CPU Cores</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{format_bytes(results.memory_gb * 1024**3)}</div>
            <div class="metric-label">Memory</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("**Storage Requirements**")
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{format_bytes(results.storage_gb * 1024**3)}</div>
            <div class="metric-label">Storage</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{format_bytes(results.local_disk_gb * 1024**3)}</div>
            <div class="metric-label">Local Disk</div>
        </div>
        """, unsafe_allow_html=True)

    # Performance estimates
    st.markdown('<div class="section-header">⚡ Performance Estimates</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Estimated Query Performance:**
        - **Max QPS**: ~{results.estimated_qps:.0f} queries/second
        - **Avg Latency**: ~{results.estimated_latency_ms:.0f}ms per query
        - **Throughput**: Suitable for {deployment_type.value} deployment
        """)

    with col2:
        if compression != CompressionType.NONE:
            savings_gb = results.memory_gb - results.memory_gb_compressed
            savings_pct = (savings_gb / results.memory_gb) * 100
            st.success(f"""
            **With {COMPRESSION_INFO[compression]['name']}:**
            - **Memory Savings**: {format_bytes(savings_gb * 1024**3)} ({savings_pct:.1f}%)
            - **Compressed Memory**: {format_bytes(results.memory_gb_compressed * 1024**3)}
            """)

    # Memory breakdown chart
    st.markdown('<div class="section-header">💾 Memory Breakdown</div>', unsafe_allow_html=True)
    
    # Create memory breakdown chart
    memory_data = {
        'Component': ['Vector Data', 'Index Structure', 'Metadata', 'GC Overhead'],
        'Memory (GB)': [
            results.vector_memory_gb,
            results.index_memory_gb,
            results.metadata_memory_gb,
            results.vector_memory_gb  # GC overhead approximation
        ],
        'Color': ['#667eea', '#764ba2', '#f093fb', '#f5576c']
    }
    
    fig = px.pie(
        memory_data, 
        values='Memory (GB)', 
        names='Component',
        color_discrete_sequence=memory_data['Color'],
        title="Memory Distribution"
    )
    fig.update_layout(
        showlegend=True,
        height=400,
        font=dict(size=12)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Deployment configuration section
    st.markdown('<div class="section-header">🚀 Deployment Configuration</div>', unsafe_allow_html=True)
    
    # Architecture recommendation
    if num_vectors < 1_000_000:
        rec_deployment = "Standalone"
        rec_desc = "Suitable for small to medium scale"
    else:
        rec_deployment = "Distributed"
        rec_desc = "Suitable for large scale"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <strong>Recommended: {rec_deployment}</strong><br>
            {rec_desc}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <strong>Note:</strong> This config is our best estimation based on experience. 
            We suggest testing with your data and traffic pattern before launching to production.
        </div>
        """, unsafe_allow_html=True)

    # Generate deployment files
    config = {
        'num_vectors': num_vectors,
        'dimensions': dimensions,
        'index_type': index_type.value,
        'compression': compression.value,
        'memory_gb': results.memory_gb,
        'cpu_cores': results.cpu_cores,
        'storage_gb': results.storage_gb,
        'deployment_type': deployment_type.value
    }

    # Download section
    st.markdown('<div class="section-header">⬇️ Install Weaviate</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🐳 Docker Compose", "☸️ Kubernetes", "📊 Configuration Summary"])
    
    with tab1:
        docker_config = generate_docker_compose(config)
        st.code(docker_config, language='yaml')
        
        st.markdown("**Quick Start:**")
        st.code("""
# Download configuration
wget -O docker-compose.yml [your-generated-config]

# Start Weaviate
docker compose up -d

# Verify installation
curl http://localhost:8080/v1/.well-known/ready
        """, language='bash')
        
        st.download_button(
            "📥 Download docker-compose.yml",
            data=docker_config,
            file_name="weaviate-docker-compose.yml",
            mime="text/yaml"
        )
    
    with tab2:
        k8s_config = generate_kubernetes_manifest(config)
        st.code(k8s_config, language='yaml')
        
        st.markdown("**Deployment Commands:**")
        st.code("""
# Apply configuration
kubectl apply -f weaviate-deployment.yaml

# Check status
kubectl get pods -l app=weaviate
kubectl get svc weaviate

# Port forward to access locally
kubectl port-forward svc/weaviate 8080:8080
        """, language='bash')
        
        st.download_button(
            "📥 Download Kubernetes YAML",
            data=k8s_config,
            file_name="weaviate-deployment.yaml",
            mime="text/yaml"
        )
    
    with tab3:
        st.markdown("### Configuration Summary")
        
        summary_data = {
            "Parameter": [
                "Number of Vectors",
                "Vector Dimensions", 
                "Index Type",
                "Compression",
                "Deployment Type",
                "Memory Required",
                "CPU Cores",
                "Storage Required",
                "Estimated QPS",
                "Estimated Latency"
            ],
            "Value": [
                f"{format_number(num_vectors)}",
                f"{dimensions}",
                INDEX_INFO[index_type]["name"],
                COMPRESSION_INFO[compression]["name"],
                deployment_type.value.title(),
                f"{results.memory_gb:.1f} GB",
                f"{results.cpu_cores:.1f}",
                f"{results.storage_gb:.1f} GB",
                f"~{results.estimated_qps:.0f}",
                f"~{results.estimated_latency_ms:.0f}ms"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.markdown("**Helm Chart Customization Examples:**")
        st.code(f"""
# Install with custom memory settings
helm install weaviate ./weaviate-chart \\
  --set resources.limits.memory={int(results.memory_gb * 1.5)}Gi \\
  --set persistence.size={int(results.storage_gb * 1.2)}Gi

# Enable ingress for external access
helm install weaviate ./weaviate-chart \\
  --set ingress.enabled=true \\
  --set ingress.hosts[0].host=weaviate.yourcompany.com

# Production deployment with monitoring
helm install weaviate ./weaviate-chart \\
  --set replicaCount=3 \\
  --set autoscaling.enabled=true \\
  --set monitoring.enabled=true \\
  --set persistence.storageClass=premium-ssd
        """, language='bash')

    # Footer with additional resources
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>📖 <strong>Resources:</strong> 
        <a href="https://weaviate.io/developers/weaviate/concepts/resources" target="_blank">Resource Planning Guide</a> | 
        <a href="https://weaviate.io/developers/weaviate/concepts/vector-indexing" target="_blank">Vector Indexing</a> | 
        <a href="https://weaviate.io/developers/weaviate/starter-guides/managing-resources/compression" target="_blank">Compression Guide</a>
        </p>
        <p>Built with ❤️ for the Weaviate community</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
