Title: Information Security Policy
Version: 2.1
Last Updated: 2024-01-10
Owner: Information Security (InfoSec)

# Information Security Policy

This document outlines the security requirements and expectations for all Meridian Technologies Inc. employees, contractors, and vendors.

## 1. Data Classification
All Meridian data must be classified into one of four categories:
* **Public:** Information intended for public consumption (e.g., marketing materials, public website content).
* **Internal:** General company information not meant for external release (e.g., employee handbook, internal memos, organization charts).
* **Confidential:** Sensitive business information (e.g., financial projections, unreleased product roadmaps, source code). Unauthorized disclosure could cause financial or reputational damage.
* **Restricted:** Highly sensitive data (e.g., Personally Identifiable Information (PII), Protected Health Information (PHI), customer payment data). Disclosure is restricted by law or could cause severe harm.

## 2. Password Requirements
All Meridian accounts must adhere to the following password policy:
* Minimum length: 16 characters.
* Must include at least one uppercase letter, one lowercase letter, one number, and one special character.
* Passwords expire every 90 days for privileged accounts (e.g., domain admins, database admins) and 180 days for standard accounts.
* Password history must prevent the reuse of the last 10 passwords.
* Passwords must not be written down or stored in unencrypted files. Use the company-approved password manager (1Password).

## 3. Multi-Factor Authentication (MFA)
MFA is mandatory for all access to Meridian networks, applications, and VPNs. Hardware security keys (YubiKey) are the preferred MFA method. Authenticator apps (e.g., Google Authenticator, Authy) are permitted as a fallback. SMS-based MFA is strictly prohibited.

## 4. Clean Desk Policy
Employees must ensure that all Confidential and Restricted information is secured when they leave their workspace, whether at the office or working remotely.
* Physical documents must be locked in a drawer or cabinet.
* Whiteboards must be erased if they contain sensitive information.
* Computer screens must be locked when unattended (Windows: Win+L, Mac: Cmd+Ctrl+Q). Screen savers must activate automatically after 10 minutes of inactivity and require a password to unlock.

## 5. Incident Reporting
All suspected security incidents (e.g., phishing attempts, lost devices, unauthorized access) must be reported immediately to the InfoSec team via the `#security-incidents` Slack channel or by emailing `security@meridiantech.com`. Do not attempt to investigate the incident yourself. Refer to the Incident Response Plan for detailed procedures.
