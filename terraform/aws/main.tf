terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }

  backend "s3" {
    bucket         = "aegis-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    kms_key_id     = "arn:aws:kms:us-east-1:123456789012:key/aegis-kms"
    dynamodb_table = "aegis-terraform-locks"
  }
}

# Multi-region configuration
locals {
  regions = {
    primary   = "us-east-1"
    secondary = "eu-west-1"
    tertiary  = "ap-southeast-1"
  }

  availability_zones = {
    "us-east-1"      = ["us-east-1a", "us-east-1b", "us-east-1c"]
    "eu-west-1"      = ["eu-west-1a", "eu-west-1b", "eu-west-1c"]
    "ap-southeast-1" = ["ap-southeast-1a", "ap-southeast-1b", "ap-southeast-1c"]
  }

  common_tags = {
    Environment  = "production"
    Project      = "aegis"
    ManagedBy    = "terraform"
    Compliance   = "soc2-hipaa-pci"
    DataPrivacy  = "gdpr-ccpa"
    CostCenter   = "engineering"
    LastUpdated  = timestamp()
  }
}

# Primary region provider
provider "aws" {
  region = local.regions.primary

  default_tags {
    tags = local.common_tags
  }
}

# Secondary region provider (EU)
provider "aws" {
  alias  = "eu"
  region = local.regions.secondary

  default_tags {
    tags = local.common_tags
  }
}

# Tertiary region provider (APAC)
provider "aws" {
  alias  = "apac"
  region = local.regions.tertiary

  default_tags {
    tags = local.common_tags
  }
}

# VPC for each region
module "vpc_primary" {
  source = "./modules/vpc"

  region             = local.regions.primary
  availability_zones = local.availability_zones[local.regions.primary]
  cidr_block        = "10.0.0.0/16"

  private_subnet_cidrs = [
    "10.0.1.0/24",
    "10.0.2.0/24",
    "10.0.3.0/24"
  ]

  public_subnet_cidrs = [
    "10.0.101.0/24",
    "10.0.102.0/24",
    "10.0.103.0/24"
  ]

  database_subnet_cidrs = [
    "10.0.201.0/24",
    "10.0.202.0/24",
    "10.0.203.0/24"
  ]

  enable_nat_gateway = true
  enable_vpn_gateway = true
  enable_flow_logs   = true

  tags = merge(local.common_tags, {
    Name = "aegis-vpc-primary"
  })
}

# EKS Cluster - Primary Region
module "eks_primary" {
  source = "./modules/eks"

  cluster_name    = "aegis-production"
  cluster_version = "1.28"
  region          = local.regions.primary
  vpc_id          = module.vpc_primary.vpc_id
  subnet_ids      = module.vpc_primary.private_subnet_ids

  # Node groups for different workloads
  node_groups = {
    # General purpose nodes
    general = {
      desired_capacity = 10
      max_capacity     = 50
      min_capacity     = 5

      instance_types = ["m6i.2xlarge", "m6a.2xlarge"]

      disk_size = 200
      disk_type = "gp3"
      disk_iops = 10000
      disk_throughput = 250

      labels = {
        role = "general"
      }

      taints = []

      tags = {
        "k8s.io/cluster-autoscaler/enabled" = "true"
        "k8s.io/cluster-autoscaler/aegis-production" = "owned"
      }
    }

    # ML workload nodes with GPU
    ml = {
      desired_capacity = 5
      max_capacity     = 20
      min_capacity     = 2

      instance_types = ["g5.4xlarge"]  # NVIDIA A10G GPUs

      disk_size = 500
      disk_type = "gp3"
      disk_iops = 16000
      disk_throughput = 1000

      labels = {
        role = "ml"
        "nvidia.com/gpu" = "true"
      }

      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NoSchedule"
        }
      ]

      tags = {
        Workload = "ml-inference"
      }
    }

    # Database nodes
    database = {
      desired_capacity = 6
      max_capacity     = 12
      min_capacity     = 3

      instance_types = ["r6i.2xlarge", "r6a.2xlarge"]  # Memory optimized

      disk_size = 1000
      disk_type = "gp3"
      disk_iops = 16000
      disk_throughput = 1000

      labels = {
        role = "database"
      }

      taints = [
        {
          key    = "dedicated"
          value  = "database"
          effect = "NoSchedule"
        }
      ]

      tags = {
        Workload = "database"
      }
    }
  }

  # OIDC for IRSA
  enable_irsa = true

  # Encryption
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks.arn
    resources        = ["secrets"]
  }

  # Logging
  cluster_enabled_log_types = [
    "api",
    "audit",
    "authenticator",
    "controllerManager",
    "scheduler"
  ]

  tags = local.common_tags
}

# RDS Aurora PostgreSQL (Multi-AZ, Multi-Region)
module "rds_aurora" {
  source = "./modules/rds-aurora"

  cluster_identifier = "aegis-postgres-primary"
  engine             = "aurora-postgresql"
  engine_version     = "15.4"

  master_username = "aegis_admin"
  database_name   = "aegis"

  vpc_id     = module.vpc_primary.vpc_id
  subnet_ids = module.vpc_primary.database_subnet_ids

  instance_class = "db.r6i.4xlarge"
  instances = {
    primary = {
      instance_class = "db.r6i.4xlarge"

    }
    replica1 = {
      instance_class = "db.r6i.4xlarge"
    }
    replica2 = {
      instance_class = "db.r6i.2xlarge"
    }
  }

  # Storage
  allocated_storage     = 1000
  max_allocated_storage = 10000
  storage_encrypted     = true
  kms_key_id           = aws_kms_key.rds.arn

  # Backup
  backup_retention_period = 35
  preferred_backup_window = "03:00-04:00"
  copy_tags_to_snapshot  = true

  # Performance Insights
  performance_insights_enabled = true
  performance_insights_retention_period = 731  # 2 years

  # Enhanced Monitoring
  enabled_cloudwatch_logs_exports = [
    "postgresql"
  ]
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring.arn

  # Global database for multi-region
  enable_global_database = true
  global_cluster_members = [
    {
      region = local.regions.secondary
      vpc_id = module.vpc_secondary.vpc_id
      subnet_ids = module.vpc_secondary.database_subnet_ids
    },
    {
      region = local.regions.tertiary
      vpc_id = module.vpc_tertiary.vpc_id
      subnet_ids = module.vpc_tertiary.database_subnet_ids
    }
  ]

  tags = local.common_tags
}

# ElastiCache Redis Cluster
module "elasticache_redis" {
  source = "./modules/elasticache"

  cluster_id           = "aegis-redis"
  engine              = "redis"
  engine_version      = "7.0"
  node_type           = "cache.r7g.2xlarge"
  num_cache_nodes     = 6  # 3 primary + 3 replicas

  vpc_id     = module.vpc_primary.vpc_id
  subnet_ids = module.vpc_primary.private_subnet_ids

  # Replication
  automatic_failover_enabled = true
  multi_az_enabled          = true

  # Security
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                = random_password.redis_auth_token.result

  # Backup
  snapshot_retention_limit = 7
  snapshot_window         = "03:00-05:00"

  # Notifications
  notification_topic_arn = aws_sns_topic.alerts.arn

  tags = local.common_tags
}

# S3 Buckets for data storage
module "s3_data" {
  source = "./modules/s3"

  buckets = {
    # ML models storage
    models = {
      bucket_name = "aegis-ml-models-${data.aws_caller_identity.current.account_id}"
      versioning  = true

      lifecycle_rules = [
        {
          id     = "archive-old-models"
          status = "Enabled"

          transition = [
            {
              days          = 90
              storage_class = "INTELLIGENT_TIERING"
            }
          ]
        }
      ]

      replication = {
        role_arn = aws_iam_role.s3_replication.arn

        rules = [
          {
            id       = "replicate-to-secondary"
            status   = "Enabled"
            priority = 1

            destination = {
              bucket = "arn:aws:s3:::aegis-ml-models-${data.aws_caller_identity.current.account_id}-eu"
              storage_class = "STANDARD_IA"
            }
          }
        ]
      }
    }

    # Audit logs storage
    audit = {
      bucket_name = "aegis-audit-logs-${data.aws_caller_identity.current.account_id}"
      versioning  = true

      lifecycle_rules = [
        {
          id     = "retain-compliance"
          status = "Enabled"

          transition = [
            {
              days          = 30
              storage_class = "STANDARD_IA"
            },
            {
              days          = 365
              storage_class = "GLACIER"
            }
          ]

          expiration = {
            days = 2555  # 7 years for compliance
          }
        }
      ]

      object_lock = {
        mode = "GOVERNANCE"
        days = 2555
      }
    }

    # Customer data with encryption
    customer_data = {
      bucket_name = "aegis-customer-data-${data.aws_caller_identity.current.account_id}"
      versioning  = true

      server_side_encryption = {
        sse_algorithm     = "aws:kms"
        kms_master_key_id = aws_kms_key.s3.arn
      }

      public_access_block = {
        block_public_acls       = true
        block_public_policy     = true
        ignore_public_acls      = true
        restrict_public_buckets = true
      }
    }
  }

  tags = local.common_tags
}

# WAF for API protection
module "waf" {
  source = "./modules/waf"

  name  = "aegis-api-waf"
  scope = "REGIONAL"

  rules = {
    # Rate limiting
    rate_limit = {
      priority = 1
      action   = "block"

      rate_based_statement = {
        limit              = 10000
        aggregate_key_type = "IP"
      }
    }

    # Geo blocking
    geo_block = {
      priority = 2
      action   = "block"

      geo_match_statement = {
        country_codes = ["CN", "RU", "KP"]  # Block high-risk countries
      }
    }

    # SQL injection protection
    sql_injection = {
      priority = 3
      action   = "block"

      managed_rule_group = {
        vendor_name = "AWS"
        name        = "AWSManagedRulesSQLiRuleSet"
      }
    }

    # Known bad inputs
    known_bad = {
      priority = 4
      action   = "block"

      managed_rule_group = {
        vendor_name = "AWS"
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
      }
    }
  }

  # Associate with ALB
  resource_arn = module.alb.arn

  tags = local.common_tags
}

# KMS Keys for encryption
resource "aws_kms_key" "eks" {
  description             = "KMS key for EKS cluster encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "aegis-eks-kms"
  })
}

resource "aws_kms_key" "rds" {
  description             = "KMS key for RDS encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = merge(local.common_tags, {
    Name = "aegis-rds-kms"
  })
}

resource "aws_kms_key" "s3" {
  description             = "KMS key for S3 encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = merge(local.common_tags, {
    Name = "aegis-s3-kms"
  })
}

# CloudWatch Alarms
module "cloudwatch_alarms" {
  source = "./modules/cloudwatch-alarms"

  alarms = {
    # API latency
    api_latency = {
      metric_name = "TargetResponseTime"
      namespace   = "AWS/ApplicationELB"
      statistic   = "Average"
      period      = 300
      threshold   = 100  # 100ms

      dimensions = {
        LoadBalancer = module.alb.arn_suffix
      }
    }

    # Error rate
    error_rate = {
      metric_name = "HTTPCode_Target_5XX_Count"
      namespace   = "AWS/ApplicationELB"
      statistic   = "Sum"
      period      = 300
      threshold   = 10

      dimensions = {
        LoadBalancer = module.alb.arn_suffix
      }
    }

    # Database connections
    db_connections = {
      metric_name = "DatabaseConnections"
      namespace   = "AWS/RDS"
      statistic   = "Average"
      period      = 300
      threshold   = 900  # 90% of max connections

      dimensions = {
        DBClusterIdentifier = module.rds_aurora.cluster_id
      }
    }

    # EKS node CPU
    eks_cpu = {
      metric_name = "node_cpu_utilization"
      namespace   = "ContainerInsights"
      statistic   = "Average"
      period      = 300
      threshold   = 80  # 80% CPU

      dimensions = {
        ClusterName = module.eks_primary.cluster_name
      }
    }
  }

  alarm_actions = [aws_sns_topic.alerts.arn]

  tags = local.common_tags
}

# SNS Topic for alerts
resource "aws_sns_topic" "alerts" {
  name = "aegis-production-alerts"

  kms_master_key_id = "alias/aws/sns"

  tags = merge(local.common_tags, {
    Name = "aegis-alerts"
  })
}

# Outputs
output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks_primary.cluster_endpoint
}

output "rds_cluster_endpoint" {
  description = "RDS cluster endpoint"
  value       = module.rds_aurora.cluster_endpoint
}

output "redis_cluster_endpoint" {
  description = "Redis cluster endpoint"
  value       = module.elasticache_redis.cluster_endpoint
}

output "api_endpoint" {
  description = "API endpoint"
  value       = "https://api.aegis-shield.ai"
}