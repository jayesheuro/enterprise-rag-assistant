Title: Incident Response Plan
Version: 3.0
Last Updated: 2024-03-12
Owner: SRE and InfoSec

# Incident Response Plan

This document defines the process for responding to service disruptions and security incidents at Meridian Technologies Inc.

## 1. Severity Levels

Incidents are classified into four severity levels based on their impact on customers and the business.

* **SEV1 (Critical):** Complete outage of a core service (e.g., production database down, main web application offline). Significant revenue impact or critical security breach (e.g., active data exfiltration). Target resolution time: < 2 hours.
* **SEV2 (High):** Major functionality broken for many customers, but the core system is still accessible. Performance severely degraded. Target resolution time: < 4 hours.
* **SEV3 (Medium):** Minor bug or functionality issue affecting a small percentage of users. No data loss risk. Workaround available. Target resolution time: < 24 hours.
* **SEV4 (Low):** Cosmetic issue, typo, or minor internal tooling problem. Target resolution time: < 7 days.

## 2. Escalation Matrix
The on-call engineer is the first point of contact for all automated alerts.

* **For SEV1/SEV2:** If the on-call engineer does not acknowledge the page within 15 minutes, it automatically escalates to the Secondary On-Call. If unacknowledged after 30 minutes, it escalates to the Engineering Manager. After 60 minutes, it escalates to the VP of Engineering.
* The Incident Commander (IC) role is assumed by the most senior engineer online until explicitly handed off. The IC is responsible for coordinating the response, not for fixing the code.

## 3. Communication Templates
Clear and timely communication is critical during an incident.

**Internal Status Update Template (Slack #incident-bridge):**
*   **Incident ID:** INC-XXXX
*   **Current Status:** [Investigating / Mitigating / Monitoring / Resolved]
*   **Impact:** [Description of what is broken and who is affected]
*   **Next Steps:** [What the team is doing right now]
*   **Next Update:** [Time in UTC]

**External Status Page Update (For SEV1/SEV2 only):**
*   "We are currently investigating reports of [issue description]. Our engineering team is actively working to identify the root cause. We will provide an update within 30 minutes."

## 4. Post-Incident Review (PIR) Process
A blameless Post-Incident Review (PIR) must be completed for every SEV1 and SEV2 incident.
* The PIR document must be drafted within 48 hours of incident resolution.
* The PIR meeting must occur within 5 business days, attended by all involved engineers and stakeholders.
* The PIR must include a timeline of events, root cause analysis (using the "5 Whys" method), and action items to prevent recurrence. Action items must be logged in Jira with a target completion date.
