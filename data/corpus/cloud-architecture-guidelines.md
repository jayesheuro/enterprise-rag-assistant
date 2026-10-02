Title: Cloud Architecture Guidelines
Version: 1.8
Last Updated: 2024-04-02
Owner: Cloud Engineering

# Cloud Architecture Standards

This document establishes the standards for all cloud infrastructure deployed at Meridian Technologies Inc.

## 1. Approved Cloud Providers
Meridian employs a multi-cloud strategy to ensure resilience and avoid vendor lock-in, with a strong primary preference.
* **Primary Provider: Amazon Web Services (AWS).** AWS is the default provider for all new applications, compute workloads, and data storage. Specifically, use region `us-east-1` for production and `us-west-2` for disaster recovery.
* **Secondary Provider: Google Cloud Platform (GCP).** GCP is approved specifically for big data workloads (BigQuery) and machine learning models. No general-purpose compute or web applications should be deployed to GCP without VP of Engineering approval.

## 2. Infrastructure as Code (IaC) Requirements
Manual provisioning of cloud resources via the AWS Management Console or GCP Console is strictly prohibited in production and staging environments.
* All infrastructure must be defined using **Terraform** (version 1.5 or higher).
* Terraform state files must be stored in the central `meridian-tf-state-prod` S3 bucket, with DynamoDB locking enabled.
* All IaC changes must go through the standard pull request review process and be applied via our CI/CD pipeline (GitHub Actions).

## 3. Tagging Standards
Every provisioned cloud resource must have the following mandatory tags. Resources lacking these tags will be automatically terminated by the cloud janitor script after 48 hours.
* `Owner`: The email address of the team responsible for the resource (e.g., `team-billing@meridiantech.com`).
* `Environment`: Must be one of `production`, `staging`, `development`, or `sandbox`.
* `Service`: The name of the microservice or application the resource belongs to.
* `DataClassification`: Must match one of the categories defined in the Information Security Policy (`Public`, `Internal`, `Confidential`, `Restricted`).

## 4. Cost Optimization
Teams are responsible for monitoring and optimizing the cost of their cloud resources.
* Rightsizing: EC2 instances and RDS databases should be scaled to maintain average CPU utilization between 40% and 70%.
* Ephemeral Environments: All `development` and `sandbox` environments must be spun down automatically outside of business hours (7 PM to 7 AM EST) and on weekends.
* Unused Resources: Unattached EBS volumes and unassociated Elastic IPs must be deleted immediately.
