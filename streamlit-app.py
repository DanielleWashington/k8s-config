import streamlit as st
import yaml
from datetime import datetime
import zipfile
import io

# Page configuration
st.set_page_config(
    page_title="EKS Cluster Generator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #130C49 0%, #61D384 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .section-header {
        color: #130C49;
        border-bottom: 2px solid #85E3BC;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .stDownloadButton button {
        background: linear-gradient(135deg, #130C49 0%, #61D384 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 Weaviate EKS Cluster Generator</h1>
    <p>Generate an EKS configuration for your Weaviate deployment</p>
</div>
""", unsafe_allow_html=True)

def generate_eksctl_config(config):
    """Generate eksctl configuration YAML"""
    
    # Build labels section
    labels_section = ""
    if config['labels']:
        labels_section = "\n      labels:\n" + "\n".join([
            f"        {key}: {value}" for key, value in config['labels'].items()
        ])
    
    # Build addons section
    addons = []
    if config['addons']['vpc_cni']:
        addons.append("""  - name: vpc-cni
    version: latest
    attachPolicyARNs:
      - arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy""")
    
    if config['addons']['core_dns']:
        addons.append("""  - name: coredns
    version: latest""")
    
    if config['addons']['kube_proxy']:
        addons.append("""  - name: kube-proxy
    version: latest""")
    
    if config['addons']['ebs_csi']:
        addons.append("""  - name: aws-ebs-csi-driver
    version: latest
    attachPolicyARNs:
      - arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy""")
    
    addons_section = ""
    if addons:
        addons_section = "\naddOns:\n" + "\n".join(addons)
    
    return f"""apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: {config['cluster_name']}
  region: {config['aws_region']}

managedNodeGroups:
  - name: {config['node_group_name']}
    instanceType: {config['instance_type']}
    desiredCapacity: {config['desired_capacity']}
    minSize: {config['min_size']}
    maxSize: {config['max_size']}
    privateNetworking: {str(config['private_networking']).lower()}
    volumeSize: {config['volume_size']}
    volumeType: {config['volume_type']}{labels_section}
    tags:
      Environment: production
      ManagedBy: eksctl
      NodeGroup: {config['node_group_name']}
    iam:
      attachPolicyARNs:
        - arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
        - arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy
        - arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
        - arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore{addons_section}"""

def generate_storage_classes():
    """Generate Kubernetes storage classes YAML"""
    return """# Storage Class for EBS volumes
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3-storage
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Delete

---
# Additional storage class for high-performance workloads
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3-high-performance
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "16000"
  throughput: "1000"
  encrypted: "true"
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Delete

---
# Storage class for io1 volumes (high IOPS)
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: io1-storage
provisioner: ebs.csi.aws.com
parameters:
  type: io1
  iops: "10000"
  encrypted: "true"
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Delete"""


# Sidebar for configuration
st.sidebar.markdown("## ⚙️ Cluster Configuration")

# Cluster basics
cluster_name = st.sidebar.text_input("Cluster Name", value="my-eks-cluster")
aws_region = st.sidebar.selectbox(
    "AWS Region",
    ["us-east-1", "us-east-2", "us-west-1", "us-west-2", 
     "eu-west-1", "eu-central-1", "ap-southeast-1", "ap-northeast-1"],
    index=3
)

st.sidebar.markdown("### 🖥️ Node Group Configuration")
node_group_name = st.sidebar.text_input("Node Group Name", value="standard-workers")

instance_type = st.sidebar.selectbox(
    "Instance Type",
    ["t3.micro", "t3.small", "t3.medium", "t3.large", "t3.xlarge",
     "m5.large", "m5.xlarge", "m5.2xlarge", "c5.large", "c5.xlarge"],
    index=2
)

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    desired_capacity = st.number_input("Desired", min_value=1, max_value=20, value=3)
with col2:
    min_size = st.number_input("Min", min_value=0, max_value=20, value=1)
with col3:
    max_size = st.number_input("Max", min_value=1, max_value=50, value=6)

col1, col2 = st.sidebar.columns(2)
with col1:
    volume_size = st.number_input("Volume Size (GB)", min_value=8, max_value=500, value=20)
with col2:
    volume_type = st.selectbox("Volume Type", ["gp3", "gp2", "io1", "io2"], index=0)

private_networking = st.sidebar.checkbox("Private Networking", value=True)

# Labels section
st.sidebar.markdown("### 🏷️ Node Labels")
if 'labels' not in st.session_state:
    st.session_state.labels = {"environment": "production"}

# Add label interface
new_key = st.sidebar.text_input("Label Key")
new_value = st.sidebar.text_input("Label Value")
if st.sidebar.button("Add Label") and new_key and new_value:
    st.session_state.labels[new_key] = new_value

# Display and manage existing labels
if st.session_state.labels:
    st.sidebar.write("**Current Labels:**")
    labels_to_remove = []
    for key, value in st.session_state.labels.items():
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.sidebar.text(f"{key}: {value}")
        with col2:
            if st.sidebar.button("❌", key=f"remove_{key}"):
                labels_to_remove.append(key)
    
    for key in labels_to_remove:
        del st.session_state.labels[key]
        st.rerun()

# Add-ons section
st.sidebar.markdown("### 🔧 Add-ons")
vpc_cni = st.sidebar.checkbox("VPC CNI", value=True)
core_dns = st.sidebar.checkbox("CoreDNS", value=True)
kube_proxy = st.sidebar.checkbox("Kube Proxy", value=True)
ebs_csi = st.sidebar.checkbox("EBS CSI Driver", value=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Configuration object
    config = {
        'cluster_name': cluster_name,
        'aws_region': aws_region,
        'node_group_name': node_group_name,
        'instance_type': instance_type,
        'desired_capacity': desired_capacity,
        'min_size': min_size,
        'max_size': max_size,
        'volume_size': volume_size,
        'volume_type': volume_type,
        'private_networking': private_networking,
        'labels': st.session_state.labels,
        'addons': {
            'vpc_cni': vpc_cni,
            'core_dns': core_dns,
            'kube_proxy': kube_proxy,
            'ebs_csi': ebs_csi
        }
    }
    
    # Preview section
    st.markdown('<h2 class="section-header">📄 Configuration Preview</h2>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📋 EKS Config", "💾 Storage Classes"])
    
    with tab1:
        eksctl_config = generate_eksctl_config(config)
        st.code(eksctl_config, language='yaml')
    
    with tab2:
        storage_config = generate_storage_classes()
        st.code(storage_config, language='yaml')

with col2:
    st.markdown('<h2 class="section-header">📥 Download Files</h2>', unsafe_allow_html=True)
    
    # Configuration summary
    st.markdown("### Configuration Summary")
    st.info(f"""
    **Cluster:** {cluster_name}  
    **Region:** {aws_region}  
    **Nodes:** {desired_capacity} × {instance_type}  
    **Storage:** {volume_size}GB {volume_type}  
    **Private:** {"✅" if private_networking else "❌"}
    """)
    
    # Generate files
    eksctl_config = generate_eksctl_config(config)
    storage_config = generate_storage_classes()
    
    # Individual downloads
    st.download_button(
        label="📄 Download EKS Config",
        data=eksctl_config,
        file_name=f"{cluster_name}-config.yaml",
        mime="text/yaml"
    )
    
    st.download_button(
        label="💾 Download Storage Classes",
        data=storage_config,
        file_name="storage-classes.yaml",
        mime="text/yaml"
    )
    
    # Create ZIP file with all configurations
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(f"{cluster_name}-config.yaml", eksctl_config)
        zip_file.writestr("storage-classes.yaml", storage_config)
    
    st.download_button(
        label="📦 Download All Files (ZIP)",
        data=zip_buffer.getvalue(),
        file_name=f"{cluster_name}-eks-config.zip",
        mime="application/zip"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    Built with ❤️ using Streamlit | Generate EKS configurations that actually work
</div>
""", unsafe_allow_html=True)