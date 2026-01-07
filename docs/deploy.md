# AWS Cloud Deployment Guide

Deploy the Florida Tax RAG system to AWS using managed databases and a single EC2 instance.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Cost Estimation](#cost-estimation)
- [Prerequisites](#prerequisites)
- [Step 1: Set Up Managed Databases](#step-1-set-up-managed-databases)
- [Step 2: Provision EC2 Instance](#step-2-provision-ec2-instance)
- [Step 3: Configure the Server](#step-3-configure-the-server)
- [Step 4: Deploy the Application](#step-4-deploy-the-application)
- [Step 5: Initialize Databases](#step-5-initialize-databases)
- [Step 6: Start the API](#step-6-start-the-api)
- [Step 7: Set Up Domain & SSL (Optional)](#step-7-set-up-domain--ssl-optional)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                              AWS Cloud                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────┐     ┌──────────────────────────────────────┐    │
│   │   Route 53   │────▶│         Application Load Balancer     │    │
│   │  (optional)  │     │              (optional)               │    │
│   └──────────────┘     └────────────────┬─────────────────────┘    │
│                                          │                          │
│                                          ▼                          │
│                        ┌─────────────────────────────────┐         │
│                        │      EC2 Instance (t3.medium)   │         │
│                        │  ┌─────────────────────────┐    │         │
│                        │  │   Docker Compose        │    │         │
│                        │  │   ┌─────────────────┐   │    │         │
│                        │  │   │  FastAPI + Nginx │   │    │         │
│                        │  │   │   (Port 80/443)  │   │    │         │
│                        │  │   └─────────────────┘   │    │         │
│                        │  └─────────────────────────┘    │         │
│                        └─────────────┬───────────────────┘         │
│                                      │                              │
│          ┌───────────────────────────┼───────────────────────┐     │
│          │                           │                       │     │
│          ▼                           ▼                       ▼     │
│  ┌───────────────┐      ┌────────────────────┐    ┌─────────────┐ │
│  │  Neo4j Aura   │      │  Weaviate Cloud    │    │ ElastiCache │ │
│  │  (Managed)    │      │  (Managed)         │    │   Redis     │ │
│  │               │      │                    │    │  (Managed)  │ │
│  └───────────────┘      └────────────────────┘    └─────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

External APIs:
  • Voyage AI (embeddings)
  • Anthropic Claude (LLM)
```

---

## Cost Estimation

### Monthly Costs (MVP/Small Scale)

| Service | Tier | Estimated Cost |
|---------|------|----------------|
| **Neo4j Aura** | Free tier | $0 |
| | Professional (if needed) | ~$65/month |
| **Weaviate Cloud** | Serverless Sandbox | $0 (14-day trial) |
| | Serverless Standard | ~$25/month |
| **AWS EC2** | t3.medium (on-demand) | ~$30/month |
| | t3.medium (reserved 1yr) | ~$19/month |
| **AWS ElastiCache** | cache.t3.micro | ~$12/month |
| **Data Transfer** | ~10GB/month | ~$1/month |
| **Route 53** | Hosted zone + queries | ~$1/month |

**Total Estimated: $40-130/month** (depending on tiers)

### API Costs (Variable)

| Service | Pricing | Est. 1000 queries/month |
|---------|---------|------------------------|
| **Voyage AI** | $0.10/1M tokens | ~$2-5 |
| **Anthropic Claude** | $3/1M input, $15/1M output | ~$15-30 |

---

## Prerequisites

### Required Accounts

1. **AWS Account** with billing enabled
2. **Neo4j Aura Account** - https://neo4j.com/cloud/aura/
3. **Weaviate Cloud Account** - https://console.weaviate.cloud/
4. **Voyage AI Account** - https://www.voyageai.com/
5. **Anthropic Account** - https://console.anthropic.com/

### Required API Keys

Before starting, obtain these API keys:

```
VOYAGE_API_KEY      # From Voyage AI dashboard
ANTHROPIC_API_KEY   # From Anthropic console
```

### Local Tools

Install on your local machine:

```bash
# AWS CLI
brew install awscli  # macOS
# or: curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"

# Configure AWS credentials
aws configure
```

---

## Step 1: Set Up Managed Databases

### 1.1 Neo4j Aura

1. Go to https://console.neo4j.io/
2. Click **Create Instance**
3. Select **AuraDB Free** (or Professional for production)
4. Choose **AWS** as cloud provider
5. Select region closest to your EC2 (e.g., `us-east-1`)
6. Click **Create**

**Save these credentials:**
```
NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<generated-password>
```

**Wait for instance to be "Running"** (2-5 minutes)

### 1.2 Weaviate Cloud

1. Go to https://console.weaviate.cloud/
2. Click **Create Cluster**
3. Select **Serverless** (Sandbox for testing, Standard for production)
4. Choose **AWS** and region matching your setup
5. Name: `florida-tax-rag`
6. Click **Create**

**Save these credentials:**
```
WEAVIATE_URL=https://<cluster-id>.weaviate.network
WEAVIATE_API_KEY=<your-api-key>  # From cluster details
```

### 1.3 AWS ElastiCache Redis

```bash
# Create a Redis cluster via AWS CLI
aws elasticache create-cache-cluster \
  --cache-cluster-id florida-tax-redis \
  --cache-node-type cache.t3.micro \
  --engine redis \
  --num-cache-nodes 1 \
  --region us-east-1

# Get the endpoint (wait a few minutes for creation)
aws elasticache describe-cache-clusters \
  --cache-cluster-id florida-tax-redis \
  --show-cache-node-info \
  --query 'CacheClusters[0].CacheNodes[0].Endpoint'
```

**Save the endpoint:**
```
REDIS_URL=redis://<endpoint>:6379/0
```

> **Note:** ElastiCache is only accessible from within your VPC. Your EC2 must be in the same VPC.

---

## Step 2: Provision EC2 Instance

### 2.1 Create Security Group

```bash
# Create security group
aws ec2 create-security-group \
  --group-name florida-tax-rag-sg \
  --description "Security group for Florida Tax RAG API"

# Allow SSH (restrict to your IP in production)
aws ec2 authorize-security-group-ingress \
  --group-name florida-tax-rag-sg \
  --protocol tcp --port 22 --cidr 0.0.0.0/0

# Allow HTTP
aws ec2 authorize-security-group-ingress \
  --group-name florida-tax-rag-sg \
  --protocol tcp --port 80 --cidr 0.0.0.0/0

# Allow HTTPS
aws ec2 authorize-security-group-ingress \
  --group-name florida-tax-rag-sg \
  --protocol tcp --port 443 --cidr 0.0.0.0/0

# Allow API port (for testing without nginx)
aws ec2 authorize-security-group-ingress \
  --group-name florida-tax-rag-sg \
  --protocol tcp --port 8000 --cidr 0.0.0.0/0
```

### 2.2 Create Key Pair

```bash
aws ec2 create-key-pair \
  --key-name florida-tax-rag-key \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/florida-tax-rag-key.pem

chmod 400 ~/.ssh/florida-tax-rag-key.pem
```

### 2.3 Launch EC2 Instance

```bash
# Launch t3.medium with Amazon Linux 2023
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \  # Amazon Linux 2023 (us-east-1)
  --instance-type t3.medium \
  --key-name florida-tax-rag-key \
  --security-groups florida-tax-rag-sg \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=florida-tax-rag}]'
```

### 2.4 Get Public IP

```bash
# Get instance ID and public IP
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=florida-tax-rag" \
  --query 'Reservations[0].Instances[0].[InstanceId,PublicIpAddress]' \
  --output text
```

**Save the public IP:**
```
EC2_PUBLIC_IP=<your-public-ip>
```

---

## Step 3: Configure the Server

### 3.1 SSH into EC2

```bash
ssh -i ~/.ssh/florida-tax-rag-key.pem ec2-user@$EC2_PUBLIC_IP
```

### 3.2 Install Docker

```bash
# Update system
sudo yum update -y

# Install Docker
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version

# IMPORTANT: Log out and back in for group changes
exit
```

### 3.3 Install Python 3.11

```bash
# SSH back in
ssh -i ~/.ssh/florida-tax-rag-key.pem ec2-user@$EC2_PUBLIC_IP

# Install Python 3.11
sudo yum install -y python3.11 python3.11-pip

# Set as default
sudo alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.11 1

# Verify
python3 --version  # Should show 3.11.x
```

### 3.4 Install Git and Clone Repository

```bash
sudo yum install -y git

# Clone the repository
git clone https://github.com/YOUR_USERNAME/florida_tax_rag.git
cd florida_tax_rag
```

---

## Step 4: Deploy the Application

### 4.1 Create Production Environment File

```bash
cat > .env << 'EOF'
# Environment
ENV=production
DEBUG=false
LOG_LEVEL=INFO

# API Keys (replace with your actual keys)
VOYAGE_API_KEY=your-voyage-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# Neo4j Aura (replace with your Aura credentials)
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password

# Weaviate Cloud (replace with your cluster URL)
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your-weaviate-api-key

# ElastiCache Redis (replace with your endpoint)
REDIS_URL=redis://your-elasticache-endpoint.cache.amazonaws.com:6379/0

# Model Configuration
LLM_MODEL=claude-sonnet-4-20250514
LLM_TEMPERATURE=0.1
EMBEDDING_MODEL=voyage-law-2

# Retrieval Configuration
RETRIEVAL_TOP_K=20
HYBRID_ALPHA=0.25
MAX_GRAPH_HOPS=2

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
EOF
```

### 4.2 Install Python Dependencies

```bash
pip3 install --user -r requirements.txt
# Or if using pyproject.toml:
pip3 install --user .
```

### 4.3 Create Production Docker Compose

Create `docker-compose.prod.yml` for API-only deployment:

```bash
cat > docker-compose.prod.yml << 'EOF'
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ENV=production
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    restart: unless-stopped
EOF
```

### 4.4 Create Dockerfile (if not exists)

```bash
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY data/processed/ ./data/processed/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
EOF
```

### 4.5 Create Nginx Configuration

```bash
mkdir -p nginx

cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    server {
        listen 80;
        server_name _;

        # Redirect HTTP to HTTPS (uncomment when SSL is configured)
        # return 301 https://$server_name$request_uri;

        location / {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Timeout settings for long queries
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 120s;
        }

        # Health check endpoint
        location /health {
            proxy_pass http://api/api/v1/health;
            proxy_set_header Host $host;
        }
    }

    # HTTPS server (uncomment when SSL is configured)
    # server {
    #     listen 443 ssl;
    #     server_name your-domain.com;
    #
    #     ssl_certificate /etc/nginx/ssl/fullchain.pem;
    #     ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    #
    #     location / {
    #         proxy_pass http://api;
    #         proxy_set_header Host $host;
    #         proxy_set_header X-Real-IP $remote_addr;
    #         proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    #         proxy_set_header X-Forwarded-Proto $scheme;
    #     }
    # }
}
EOF
```

---

## Step 5: Initialize Databases

### 5.1 Verify Connectivity

```bash
# Test Neo4j connection
python3 -c "
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
)
with driver.session() as session:
    result = session.run('RETURN 1')
    print('Neo4j connected:', result.single()[0])
driver.close()
"

# Test Weaviate connection
python3 -c "
import weaviate
import os
from dotenv import load_dotenv
load_dotenv()
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv('WEAVIATE_URL'),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv('WEAVIATE_API_KEY'))
)
print('Weaviate connected:', client.is_ready())
client.close()
"

# Test Redis connection (if accessible)
python3 -c "
import redis
import os
from dotenv import load_dotenv
load_dotenv()
r = redis.from_url(os.getenv('REDIS_URL'))
print('Redis connected:', r.ping())
"
```

### 5.2 Initialize Neo4j Schema

```bash
python3 scripts/init_neo4j.py --verify
```

**Expected output:**
```
Neo4j Schema:
  Constraints: 2
  Indexes: 6
Loading documents...
  Documents: 1,152
  Chunks: 3,022
```

### 5.3 Initialize Weaviate Collection

```bash
python3 scripts/init_weaviate.py --verify
```

### 5.4 Generate Embeddings

This step calls the Voyage AI API and may take 10-20 minutes:

```bash
# Generate embeddings (cached in Redis)
python3 scripts/generate_embeddings.py --verify

# Monitor progress
# Progress: 3022/3022 chunks
```

### 5.5 Load Embeddings into Weaviate

```bash
python3 scripts/load_weaviate.py
python3 scripts/verify_vector_store.py
```

---

## Step 6: Start the API

### 6.1 Build and Start

```bash
# Build the Docker image
docker-compose -f docker-compose.prod.yml build

# Start the services
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps
```

### 6.2 Verify Deployment

```bash
# Health check
curl http://localhost/health

# Or directly to API
curl http://localhost:8000/

# Test a query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Florida sales tax rate?"}'
```

### 6.3 View Logs

```bash
# All logs
docker-compose -f docker-compose.prod.yml logs -f

# API logs only
docker-compose -f docker-compose.prod.yml logs -f api
```

### 6.4 Set Up Auto-Start on Boot

```bash
# Create systemd service
sudo tee /etc/systemd/system/florida-tax-rag.service << 'EOF'
[Unit]
Description=Florida Tax RAG API
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ec2-user/florida_tax_rag
ExecStart=/usr/local/bin/docker-compose -f docker-compose.prod.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.prod.yml down
User=ec2-user

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl enable florida-tax-rag
sudo systemctl start florida-tax-rag
```

---

## Step 7: Set Up Domain & SSL (Optional)

### 7.1 Register Domain in Route 53

```bash
# Create hosted zone (if using Route 53 for DNS)
aws route53 create-hosted-zone \
  --name your-domain.com \
  --caller-reference $(date +%s)
```

### 7.2 Create A Record

```bash
# Point domain to EC2
aws route53 change-resource-record-sets \
  --hosted-zone-id YOUR_HOSTED_ZONE_ID \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "api.your-domain.com",
        "Type": "A",
        "TTL": 300,
        "ResourceRecords": [{"Value": "YOUR_EC2_PUBLIC_IP"}]
      }
    }]
  }'
```

### 7.3 Install SSL with Let's Encrypt

```bash
# SSH into EC2
ssh -i ~/.ssh/florida-tax-rag-key.pem ec2-user@$EC2_PUBLIC_IP

# Install certbot
sudo yum install -y certbot

# Stop nginx temporarily
docker-compose -f docker-compose.prod.yml stop nginx

# Get certificate
sudo certbot certonly --standalone -d api.your-domain.com

# Copy certificates
mkdir -p nginx/ssl
sudo cp /etc/letsencrypt/live/api.your-domain.com/fullchain.pem nginx/ssl/
sudo cp /etc/letsencrypt/live/api.your-domain.com/privkey.pem nginx/ssl/
sudo chown -R ec2-user:ec2-user nginx/ssl/

# Update nginx.conf to enable HTTPS (uncomment the HTTPS server block)

# Restart nginx
docker-compose -f docker-compose.prod.yml up -d nginx
```

### 7.4 Auto-Renew Certificates

```bash
# Add cron job for renewal
echo "0 0 1 * * certbot renew --quiet && docker-compose -f /home/ec2-user/florida_tax_rag/docker-compose.prod.yml restart nginx" | sudo tee -a /etc/crontab
```

---

## Monitoring & Maintenance

### Health Monitoring

```bash
# Quick health check script
cat > /home/ec2-user/check_health.sh << 'EOF'
#!/bin/bash
HEALTH=$(curl -s http://localhost/health)
STATUS=$(echo $HEALTH | jq -r '.status')

if [ "$STATUS" != "healthy" ]; then
    echo "ALERT: API unhealthy - $HEALTH"
    # Add notification here (SNS, email, etc.)
fi
EOF
chmod +x /home/ec2-user/check_health.sh

# Add to crontab (every 5 minutes)
echo "*/5 * * * * /home/ec2-user/check_health.sh >> /var/log/health_check.log 2>&1" | crontab -
```

### CloudWatch Logs (Optional)

```bash
# Install CloudWatch agent
sudo yum install -y amazon-cloudwatch-agent

# Configure to send Docker logs to CloudWatch
sudo tee /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'EOF'
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/lib/docker/containers/*/*.log",
            "log_group_name": "florida-tax-rag",
            "log_stream_name": "{instance_id}/docker"
          }
        ]
      }
    }
  }
}
EOF

sudo systemctl start amazon-cloudwatch-agent
```

### Backup Strategy

**Neo4j Aura**: Automatic daily backups (included in Aura)

**Weaviate Cloud**: Automatic backups (included in managed service)

**Application Data**:
```bash
# Backup processed data and embeddings
tar -czvf backup-$(date +%Y%m%d).tar.gz data/processed/ .env

# Upload to S3
aws s3 cp backup-$(date +%Y%m%d).tar.gz s3://your-backup-bucket/
```

### Updating the Application

```bash
# SSH into EC2
ssh -i ~/.ssh/florida-tax-rag-key.pem ec2-user@$EC2_PUBLIC_IP
cd florida_tax_rag

# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Verify
curl http://localhost/health
```

---

## Troubleshooting

### Connection Issues

**Neo4j Aura connection failed:**
```bash
# Check if URI uses neo4j+s:// (TLS required for Aura)
# Verify password doesn't have special characters that need escaping
python3 -c "from neo4j import GraphDatabase; print('Testing...')"
```

**Weaviate Cloud connection failed:**
```bash
# Verify API key is correct
# Check if cluster is in "Ready" state in Weaviate console
curl -H "Authorization: Bearer $WEAVIATE_API_KEY" "$WEAVIATE_URL/v1/.well-known/ready"
```

**ElastiCache connection failed:**
```bash
# ElastiCache is VPC-only - EC2 must be in same VPC
# Check security group allows port 6379 from EC2
aws ec2 describe-security-groups --group-ids sg-xxxxx
```

### API Not Starting

```bash
# Check Docker logs
docker-compose -f docker-compose.prod.yml logs api

# Common issues:
# - Missing environment variables
# - Database connection failures
# - Port already in use

# Check if port is in use
sudo lsof -i :8000
```

### High Latency

```bash
# Check API metrics
curl http://localhost:8000/api/v1/metrics

# Common causes:
# - Cold start (first query after restart)
# - Embedding generation (check Redis cache hits)
# - Large result sets (reduce RETRIEVAL_TOP_K)
```

### Out of Memory

```bash
# Check memory usage
free -h
docker stats

# If OOM, consider:
# - Upgrading to t3.large (8GB RAM)
# - Reducing worker count in Dockerfile
# - Adding swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Quick Reference

### Important URLs

| Service | URL |
|---------|-----|
| API | `http://YOUR_EC2_IP:8000` |
| API Docs | `http://YOUR_EC2_IP:8000/docs` |
| Health Check | `http://YOUR_EC2_IP:8000/api/v1/health` |
| Neo4j Aura Console | https://console.neo4j.io |
| Weaviate Console | https://console.weaviate.cloud |

### Useful Commands

```bash
# Start services
docker-compose -f docker-compose.prod.yml up -d

# Stop services
docker-compose -f docker-compose.prod.yml down

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Restart API
docker-compose -f docker-compose.prod.yml restart api

# Check health
curl http://localhost:8000/api/v1/health

# Test query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the sales tax rate?"}'
```

---

## See Also

- [Local Development](./development.md)
- [Configuration Reference](./configuration.md)
- [API Documentation](./api.md)
- [Troubleshooting Guide](./troubleshooting.md)
