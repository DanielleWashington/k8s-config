#!/bin/bash

CONFIG_FILE=${1:-"configs/sample-cluster.yaml"}
OUTPUT_DIR="generated"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Generating EKS configuration files..."

CLUSTER_NAME=$(yq eval '.cluster.name' "$CONFIG_FILE")

gomplate -d config="$CONFIG_FILE" \
         -f templates/eksctl-config.yaml.tmpl \
         -o "$OUTPUT_DIR/${CLUSTER_NAME}-config.yaml"

gomplate -d config="$CONFIG_FILE" \
         -f templates/storage-class.yaml.tmpl \
         -o "$OUTPUT_DIR/storage-classes.yaml"

gomplate -d config="$CONFIG_FILE" \
         -f templates/deploy-commands.sh.tmpl \
         -o "$OUTPUT_DIR/deploy-${CLUSTER_NAME}.sh"

chmod +x "$OUTPUT_DIR/deploy-${CLUSTER_NAME}.sh"

echo "✅ Generated files:"
echo "  • $OUTPUT_DIR/${CLUSTER_NAME}-config.yaml"
echo "  • $OUTPUT_DIR/storage-classes.yaml"
echo "  • $OUTPUT_DIR/deploy-${CLUSTER_NAME}.sh"
echo ""
echo "🚀 To deploy:"
echo "  cd $OUTPUT_DIR && ./deploy-${CLUSTER_NAME}.sh"