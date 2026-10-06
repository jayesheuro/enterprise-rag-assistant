import json
from pathlib import Path

questions = [
    # Factual
    {"question_id": "q001", "question": "What is the deadline for drafting a Post-Incident Review (PIR) after an incident is resolved?", "ground_truth": "A PIR draft must be completed within 48 hours of resolution for SEV1 and SEV2 incidents.", "expected_source_docs": ["incident-response-plan.md"], "question_type": "factual", "created_by": "manual"},
    {"question_id": "q002", "question": "How many days of PTO does an employee with 3 years of tenure receive?", "ground_truth": "Employees with 0-5 years of tenure receive 20 days of PTO per calendar year.", "expected_source_docs": ["employee-handbook.md"], "question_type": "factual", "created_by": "manual"},
    {"question_id": "q003", "question": "What is the minimum required password length?", "ground_truth": "The minimum password length is 16 characters.", "expected_source_docs": ["information-security-policy.md"], "question_type": "factual", "created_by": "manual"},
    {"question_id": "q004", "question": "Are employees allowed to access source code on their personal BYOD devices?", "ground_truth": "No, accessing source code on personal devices is strictly prohibited.", "expected_source_docs": ["acceptable-use-policy.md"], "question_type": "factual", "created_by": "manual"},
    {"question_id": "q005", "question": "How much advance notice must be provided before deprecating an API?", "ground_truth": "A minimum of 6 months notice is required before shutting off a deprecated API.", "expected_source_docs": ["api-design-standards.md"], "question_type": "factual", "created_by": "manual"},
    {"question_id": "q006", "question": "What is the minimum Terraform version required for Infrastructure as Code?", "ground_truth": "Terraform version 1.5 or higher must be used.", "expected_source_docs": ["cloud-architecture-guidelines.md"], "question_type": "factual", "created_by": "manual"},
    {"question_id": "q007", "question": "What is the maximum allowed size for a pull request?", "ground_truth": "A pull request can be a maximum of 400 lines of code, excluding auto-generated or lock files.", "expected_source_docs": ["code-review-guidelines.txt"], "question_type": "factual", "created_by": "manual"},
    {"question_id": "q008", "question": "How long must financial records be retained?", "ground_truth": "Financial records must be retained for 10 years.", "expected_source_docs": ["data-retention-policy.md"], "question_type": "factual", "created_by": "manual"},
    {"question_id": "q009", "question": "What are the Recovery Time Objective (RTO) and Recovery Point Objective (RPO) for Tier 1 systems?", "ground_truth": "Tier 1 systems have an RTO of 4 hours and an RPO of 15 minutes.", "expected_source_docs": ["disaster-recovery-plan.txt"], "question_type": "factual", "created_by": "manual"},
    {"question_id": "q010", "question": "Who must approve a pull request that modifies a database schema?", "ground_truth": "At least one approval must be from a Staff Engineer or higher.", "expected_source_docs": ["code-review-guidelines.txt"], "question_type": "factual", "created_by": "manual"},
    {"question_id": "q011", "question": "After how many minutes of inactivity must a screen lock automatically activate?", "ground_truth": "Screen savers must activate automatically after 10 minutes of inactivity.", "expected_source_docs": ["information-security-policy.md"], "question_type": "factual", "created_by": "manual"},
    {"question_id": "q012", "question": "What time does the HR Welcome Orientation start on a new hire's first day?", "ground_truth": "The HR Welcome Orientation starts at 9:00 AM.", "expected_source_docs": ["onboarding-guide.md"], "question_type": "factual", "created_by": "manual"},
    {"question_id": "q013", "question": "What is the rate limit for paginated public APIs?", "ground_truth": "Paginated endpoints are limited to 50 requests per minute.", "expected_source_docs": ["api-design-standards.md"], "question_type": "factual", "created_by": "manual"},
    {"question_id": "q014", "question": "In which AWS region is the Disaster Recovery environment located?", "ground_truth": "The Disaster Recovery region is us-west-2.", "expected_source_docs": ["cloud-architecture-guidelines.md", "disaster-recovery-plan.txt"], "question_type": "factual", "created_by": "manual"},
    {"question_id": "q015", "question": "How must physical paper documents containing confidential data be destroyed?", "ground_truth": "Paper documents must be cross-cut shredded.", "expected_source_docs": ["data-retention-policy.md"], "question_type": "factual", "created_by": "manual"},
    
    # Multi-hop
    {"question_id": "q016", "question": "If I am modifying a core database schema, how many approvals do I need on my PR, and what level of engineer must provide one of them?", "ground_truth": "You need at least two approvals from engineers on the owning team, and because it modifies a core database schema, at least one approval must be from a Staff Engineer or higher.", "expected_source_docs": ["code-review-guidelines.txt"], "question_type": "multi_hop", "created_by": "manual"},
    {"question_id": "q017", "question": "Can I manually provision a new EC2 instance in us-east-1 using the AWS Console, and what tags would it need?", "ground_truth": "No, manual console provisioning is strictly prohibited in production/staging; you must use Terraform. Any resources would need 4 tags: Owner, Environment, Service, and DataClassification.", "expected_source_docs": ["cloud-architecture-guidelines.md"], "question_type": "multi_hop", "created_by": "manual"},
    {"question_id": "q018", "question": "For a SEV1 incident, what is the target resolution time, and when is the PIR draft due?", "ground_truth": "The target resolution time for a SEV1 incident is under 2 hours, and the PIR draft must be completed within 48 hours of resolution.", "expected_source_docs": ["incident-response-plan.md"], "question_type": "multi_hop", "created_by": "manual"},
    {"question_id": "q019", "question": "Can I set up SMS-based MFA for my Okta account during my Day 1 onboarding?", "ground_truth": "No, SMS-based MFA is strictly prohibited. You should use a YubiKey or an authenticator app.", "expected_source_docs": ["onboarding-guide.md", "information-security-policy.md"], "question_type": "multi_hop", "created_by": "manual"},
    {"question_id": "q020", "question": "If we terminate a contract with a customer today, how long do we keep their PII, and what standard is used to securely erase the digital data?", "ground_truth": "Customer PII is retained for 7 years post-termination. Digital data must be securely erased following NIST 800-88 guidelines.", "expected_source_docs": ["data-retention-policy.md"], "question_type": "multi_hop", "created_by": "manual"},
    
    # Out of scope
    {"question_id": "q021", "question": "What was the company's annual revenue for the year 2023?", "ground_truth": "", "expected_source_docs": [], "question_type": "out_of_scope", "created_by": "manual"},
    {"question_id": "q022", "question": "How do I configure the Cisco VPN client for remote access on my personal laptop?", "ground_truth": "", "expected_source_docs": [], "question_type": "out_of_scope", "created_by": "manual"},
    {"question_id": "q023", "question": "What are the names of the current board members of Meridian Technologies?", "ground_truth": "", "expected_source_docs": [], "question_type": "out_of_scope", "created_by": "manual"},
    {"question_id": "q024", "question": "How many vacation days do part-time contractors receive per year?", "ground_truth": "", "expected_source_docs": [], "question_type": "out_of_scope", "created_by": "manual"},
    {"question_id": "q025", "question": "Can you explain the detailed source code architecture of the main billing module?", "ground_truth": "", "expected_source_docs": [], "question_type": "out_of_scope", "created_by": "manual"},
    
    # Ambiguous
    {"question_id": "q026", "question": "How do I request time off?", "ground_truth": "The documents mention that you receive a certain number of PTO days based on tenure, but they do not specify the exact system or process for requesting time off.", "expected_source_docs": ["employee-handbook.md"], "question_type": "ambiguous", "created_by": "manual"},
    {"question_id": "q027", "question": "Who is the incident commander?", "ground_truth": "The Incident Commander (IC) role is assigned to the senior-most engineer online during an incident, but the documents do not name a specific person.", "expected_source_docs": ["incident-response-plan.md"], "question_type": "ambiguous", "created_by": "manual"},
    {"question_id": "q028", "question": "What should I do if I lose my device?", "ground_truth": "If you lose a company device, you must report it to IT and InfoSec within 24 hours.", "expected_source_docs": ["acceptable-use-policy.md"], "question_type": "ambiguous", "created_by": "manual"},
    {"question_id": "q029", "question": "How long do we keep data before deleting it?", "ground_truth": "Data retention periods vary by data type: 7 years post-termination for customer data, 10 years for financial records, 5 years post-departure for employee records, and 90 days hot/365 days cold for system logs.", "expected_source_docs": ["data-retention-policy.md"], "question_type": "ambiguous", "created_by": "manual"},
    {"question_id": "q030", "question": "When are we allowed to wear casual clothes to work?", "ground_truth": "Casual attire is acceptable for typical days in the office, but business casual or business formal is required if meeting with clients or external partners.", "expected_source_docs": ["employee-handbook.md"], "question_type": "ambiguous", "created_by": "manual"},
]

out_path = Path("eval/golden/golden_v1.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

# Also create other required directories
for d in ["eval/reports", "docs/architecture", "docs/learning", "docs/question-bank"]:
    Path(d).mkdir(parents=True, exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    for q in questions:
        f.write(json.dumps(q) + "\n")
print(f"Created golden dataset at {out_path}")

