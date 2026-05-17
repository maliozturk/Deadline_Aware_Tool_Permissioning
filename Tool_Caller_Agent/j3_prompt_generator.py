# =============================================================================
#  FIRM-DEADLINE TOOL CONTROL (FTC)
#  Product Signature: FTC
# ------------------------------------------------------------------------------
#  File: Tool_Caller_Agent/j3_prompt_generator.py
#  Purpose: Generate cyber-IR prompt set and combined prompt set for J=3.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-05-17
#  Environment: Python 3.11
# =============================================================================

"""Generate the cyber-IR prompt set and combined prompt set.

Outputs:
  - Tool_Caller_Agent/j3_prompt_set_cyber_ir.jsonl
  - Tool_Caller_Agent/j3_prompt_set_combined.jsonl
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# ── Simple prompts (30%: ~50 prompts) ─────────────────────────────────────

SIMPLE_PROMPTS = [
    "What is the IP address 198.51.100.42 associated with in our threat intel?",
    "Is the domain evil-update.example.net flagged in our IOC database?",
    "What malware family is linked to SHA-256 hash a1b2c3d4e5f6?",
    "What is the MITRE ATT&CK technique for credential dumping?",
    "What port does SSH typically run on?",
    "What is the CVE number for the Log4Shell vulnerability?",
    "What is the default SNMP community string?",
    "Which MITRE ATT&CK tactic covers initial access?",
    "What is a C2 beacon in the context of malware analysis?",
    "What file extension do YARA rules typically use?",
    "What is the difference between a vulnerability and an exploit?",
    "What protocol does DNS use by default?",
    "What is the standard port for HTTPS?",
    "Define 'lateral movement' in cybersecurity.",
    "What does the acronym IOC stand for in threat intelligence?",
    "What is a zero-day vulnerability?",
    "What is the purpose of a SOC in an organization?",
    "What is the default port for RDP?",
    "What does EDR stand for?",
    "What is phishing?",
    "What is the purpose of a SIEM system?",
    "What is a botnet?",
    "Define 'threat actor' in cybersecurity.",
    "What is the CVSS score range?",
    "What is a honeypot?",
    "What does TTP stand for in threat intelligence?",
    "What is the purpose of network segmentation?",
    "What is a watering hole attack?",
    "What is the difference between IDS and IPS?",
    "What is DLP in cybersecurity?",
    "What is a reverse shell?",
    "What tool is commonly used for packet capture?",
    "What is the purpose of a firewall rule?",
    "Define 'exfiltration' in a data breach context.",
    "What is a brute force attack?",
    "What is multi-factor authentication?",
    "What is the default port for SMB?",
    "What is an APT group?",
    "What is ransomware?",
    "What does SOC Level 1 analyst typically handle?",
    "What is a sandbox in malware analysis?",
    "What is OSINT?",
    "What is the purpose of a VPN?",
    "What is a DDoS attack?",
    "What is a rootkit?",
    "What does SOAR stand for?",
    "What is an indicator of compromise?",
    "What is the kill chain model?",
    "What is privilege escalation?",
    "What is threat hunting?",
]

# ── Medium prompts (40%: ~65 prompts) ──────────────────────────────────────

MEDIUM_PROMPTS = [
    "Investigate a sudden spike in failed SSH logins on host web-prod-04 between 03:00 and 04:00 UTC.",
    "Determine whether IOC 198.51.100.42 warrants a containment action given current threat-intel context.",
    "An endpoint reports a new outbound connection to a domain flagged yesterday in threat intel. What is the recommended next step?",
    "A user reports receiving a suspicious email with a PDF attachment. What triage steps should the SOC take?",
    "Correlate the following: EDR alert for PowerShell execution on WSRV-12, and DNS query to evil-update.example.net from the same host.",
    "Check if any runbook covers the containment of a compromised endpoint detected via Cobalt Strike beacon.",
    "A new YARA rule triggered on a file uploaded to the malware sandbox. What follow-up actions are needed?",
    "Assess whether the SSL certificate anomaly reported by the proxy is related to a known MITM campaign.",
    "A host in the DMZ is sending unusually large DNS responses. Could this indicate DNS tunneling exfiltration?",
    "Review the IOC entry for domain evil-update.example.net and recommend blocking actions.",
    "What containment steps should be taken for a host exhibiting lateral movement behavior?",
    "A DLP alert fired for a large file upload to a personal cloud storage service. What investigation steps apply?",
    "Correlate a phishing email header analysis with known threat actor infrastructure patterns.",
    "Determine if the failed login pattern on the VPN gateway matches known credential stuffing campaigns.",
    "Assess the risk of a new outbound connection to an IP in our threat intel watchlist.",
    "A security scanner detected an unpatched Apache server in the DMZ. What is the remediation priority?",
    "Check if the file hash detected by the EDR matches any known malware families in our IOC database.",
    "Investigate why a production server is making outbound connections to a Tor exit node.",
    "Review the runbook for DDoS mitigation and identify applicable steps for the current attack vector.",
    "A host is generating ICMP traffic at 10x normal volume. What could this indicate?",
    "Correlate firewall deny logs with the IOC list to identify potential blocked C2 attempts.",
    "A user account was locked after 50 failed password attempts in 5 minutes. Is this a targeted attack?",
    "What is the recommended procedure for isolating a server suspected of running a crypto miner?",
    "Assess whether a newly discovered open S3 bucket poses an immediate data leak risk.",
    "Investigate a spike in outbound SMTP traffic from an internal mail relay.",
    "Determine if the executable flagged by antivirus is a false positive or a genuine threat based on IOC data.",
    "A network sensor detected port scanning from an internal host. What escalation path applies?",
    "Review the containment runbook and determine if credential rotation is required for the compromised service account.",
    "A WAF alert shows SQL injection attempts against the customer portal. What immediate actions are needed?",
    "Correlate VPN login anomalies with geo-impossible travel alerts for user jsmith@corp.",
    "Assess the threat level of a USB device insertion alert on a classified workstation.",
    "A third-party vendor reported a breach. What steps should our SOC take to assess exposure?",
    "Investigate alerts for multiple hosts beaconing to the same external IP at regular intervals.",
    "Determine if the anomalous registry modification on host WSRV-15 matches known persistence techniques.",
    "A certificate transparency log shows a new cert issued for our domain by an unknown CA. What should we do?",
    "Correlate the IDS signature alert with recent threat intel on the targeted vulnerability.",
    "Investigate an alert for unusual scheduled task creation across multiple domain-joined hosts.",
    "Assess whether a detected LSASS memory access pattern indicates credential harvesting.",
    "A SOC analyst reports a possible insider threat. What data sources should be reviewed first?",
    "Determine the blast radius of a compromised service account with domain admin privileges.",
    "Investigate DNS queries to a recently registered domain from multiple internal hosts.",
    "A web application firewall detected a rapid increase in 403 errors. Is this reconnaissance?",
    "Correlate endpoint telemetry showing PowerShell download cradle execution with threat intel.",
    "Assess the impact of a misconfigured cloud security group allowing unrestricted SSH access.",
    "Investigate a report of data exfiltration through steganography in uploaded images.",
    "Determine if the spike in Kerberos ticket requests indicates a golden ticket attack.",
    "Review the incident response runbook for steps to handle a supply chain compromise.",
    "A host is making API calls to an internal service it should not have access to. What investigation steps apply?",
    "Correlate a phishing campaign targeting our executives with known spear-phishing threat actors.",
    "Assess the risk of a deprecated TLS 1.0 connection detected on a payment processing server.",
    "Investigate a SOC alert for process hollowing detected on a workstation.",
    "Determine if the unusual increase in email forwarding rules indicates account compromise.",
    "Review firewall logs for evidence of data exfiltration over DNS to a known C2 domain.",
    "A new vulnerability in our VPN appliance was published. Assess our exposure and prioritize patching.",
    "Investigate an anomalous spike in authentication events from a service account during off-hours.",
    "Correlate web proxy logs showing connections to paste sites with potential data leak indicators.",
    "Assess whether the detected WMI activity on multiple hosts is legitimate admin work or malicious.",
    "A threat intel report mentions our industry vertical as a target. What proactive measures should we take?",
    "Determine the appropriate escalation path for a ransomware negotiation request received via email.",
    "Investigate a SIEM correlation rule that triggered for simultaneous logins from two countries.",
    "Assess the effectiveness of our current email filtering rules against the latest phishing techniques.",
    "Review the triage workflow for a malware alert that triggered during a software deployment.",
    "Correlate network flow data with endpoint logs to trace the lateral movement path of an attacker.",
    "Determine if the new scheduled task on the domain controller is part of a persistence mechanism.",
    "Investigate a sudden drop in SIEM event ingestion rate from a critical log source.",
]

# ── Complex prompts (30%: ~50 prompts) ─────────────────────────────────────

COMPLEX_PROMPTS = [
    "Summarize the likely root cause of incident ticket INC-4521 using available runbook excerpts.",
    "Conduct a multi-step investigation of a suspected APT intrusion: correlate EDR alerts, DNS anomalies, and IOC hits to build a timeline.",
    "Given IOC hits for both IP 198.51.100.42 and domain evil-update.example.net on the same host, determine if this is a coordinated campaign.",
    "Reconstruct the attack chain for a compromised endpoint: initial access via phishing, credential theft via Mimikatz, lateral movement to domain controller.",
    "Assess the organization's exposure to the Midnight Typhoon APT based on current IOC data and defensive posture.",
    "Investigate a potential supply chain compromise: trace the attack from the compromised npm package to affected internal systems.",
    "Determine whether the LockByte ransomware campaign poses an imminent threat given our industry and current defensive controls.",
    "Correlate multiple data sources (EDR, SIEM, DNS, proxy logs) to identify the full scope of a credential harvesting campaign.",
    "Evaluate the effectiveness of our current detection rules against the MITRE ATT&CK techniques used by Midnight Typhoon.",
    "Investigate a multi-stage attack: initial spear-phish, macro-enabled document execution, C2 establishment, and data exfiltration.",
    "Build a threat model for our public-facing web infrastructure based on current threat intel and recent attack patterns.",
    "Assess the risk of insider data exfiltration given the behavioral indicators described in our threat intel notes.",
    "Investigate lateral movement across three network segments, correlating firewall logs, authentication events, and endpoint telemetry.",
    "Evaluate whether our DDoS mitigation strategy is adequate based on the attack patterns described in our runbooks.",
    "Conduct a gap analysis of our incident response capabilities against the TTPs of the Midnight Typhoon APT group.",
    "Investigate a complex phishing campaign: analyze email headers, trace the infrastructure, and correlate with known threat actors.",
    "Assess the blast radius of a compromised domain admin account and recommend containment actions using available runbooks.",
    "Determine if multiple seemingly unrelated alerts (DNS tunneling, certificate anomaly, PowerShell execution) are part of a single campaign.",
    "Evaluate the sufficiency of our YARA rules against the latest malware samples documented in our IOC entries.",
    "Investigate a suspected ransomware pre-deployment stage: identify encryption staging, C2 communication, and lateral movement indicators.",
    "Conduct a post-incident review of a phishing-initiated breach using all available runbook and threat intel documentation.",
    "Assess the organization's vulnerability to a supply chain attack based on our current vendor risk posture and available threat intel.",
    "Investigate a multi-vector attack combining DDoS distraction with simultaneous data exfiltration through DNS tunneling.",
    "Evaluate the correlation between a recent certificate transparency anomaly and known MITM campaigns in our threat intel.",
    "Build an investigation timeline for a suspected APT presence, from initial beacon detection to confirmed lateral movement.",
    "Assess the potential impact of a compromised backup system and recommend recovery procedures using available runbooks.",
    "Investigate whether anomalous VPN access patterns indicate a compromised credential or a legitimate remote worker.",
    "Conduct a threat hunt for indicators of the LockByte ransomware affiliate model across our endpoint fleet.",
    "Evaluate whether our containment procedures are adequate for a scenario involving simultaneous compromise of multiple segments.",
    "Investigate a complex insider threat case: correlate DLP alerts, access logs, and behavioral analytics to build a case.",
    "Assess the strategic implications of two concurrent APT campaigns targeting our organization from different threat actors.",
    "Determine the full kill chain of an attack that began with a watering hole compromise of an industry news site.",
    "Evaluate our detection and response capabilities against DNS tunneling exfiltration techniques documented in our IOC database.",
    "Investigate a potential zero-day exploitation: correlate crash dumps, anomalous behavior, and threat intel to assess severity.",
    "Build a comprehensive risk assessment for a newly discovered vulnerability affecting our core infrastructure.",
    "Conduct a red-team-style analysis of our external attack surface using documented IOCs and threat intel.",
    "Investigate a suspected state-sponsored attack: correlate TTPs with known nation-state threat actors in our database.",
    "Assess the effectiveness of our layered defense strategy against the specific attack vectors described in recent threat intel.",
    "Determine the optimal containment strategy for a worm-like malware spreading through SMB across our internal network.",
    "Evaluate the forensic evidence preservation requirements for a potential law enforcement referral.",
    "Investigate a complex cloud security incident involving misconfigured IAM roles, unauthorized API calls, and data exposure.",
    "Build an evidence-based recommendation for executive leadership on whether to pay a ransomware demand.",
    "Assess the completeness of our incident response runbooks against the NIST Cybersecurity Framework.",
    "Investigate a potential data breach involving encrypted exfiltration channels and determine the scope of data loss.",
    "Evaluate whether our current threat hunting capabilities can detect the advanced evasion techniques described in recent intel.",
    "Conduct a comprehensive investigation of a business email compromise (BEC) spanning multiple departments and vendors.",
    "Assess the organization's resilience to a destructive wiper malware attack based on backup and recovery capabilities.",
    "Investigate a suspected firmware-level compromise on network equipment and determine containment priorities.",
    "Build a strategic threat assessment combining all available IOC data, threat intel, and runbook gap analysis.",
    "Evaluate the end-to-end security of our software development pipeline against supply chain attack vectors.",
]


def _build_cyber_ir_prompts():
    """Build the cyber-IR JSONL prompt set."""
    prompts = []
    pid = 1
    for text in SIMPLE_PROMPTS:
        prompts.append({
            "prompt_id": f"cir_{pid:04d}",
            "domain": "cyber_ir",
            "complexity": "simple",
            "text": text,
        })
        pid += 1
    for text in MEDIUM_PROMPTS:
        prompts.append({
            "prompt_id": f"cir_{pid:04d}",
            "domain": "cyber_ir",
            "complexity": "medium",
            "text": text,
        })
        pid += 1
    for text in COMPLEX_PROMPTS:
        prompts.append({
            "prompt_id": f"cir_{pid:04d}",
            "domain": "cyber_ir",
            "complexity": "complex",
            "text": text,
        })
        pid += 1
    return prompts


def _load_legacy_prompts():
    """Extract prompts from the existing Agent_V3.py inline lists."""
    # Import the inline prompt lists from Agent_V3
    agent_path = os.path.join(SCRIPT_DIR, "Agent_V3.py")
    # We parse the lists directly rather than importing to avoid side effects
    simple_prompts = []
    complex_prompts = []

    with open(agent_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract simple_prompts_en
    import ast
    # Find the list boundaries
    s_start = content.find("simple_prompts_en = [")
    s_end = content.find("]", content.find("]", s_start) + 1)
    # Actually, let's use a more robust approach
    # Just import the module variables
    import importlib.util
    spec = importlib.util.spec_from_file_location("agent_v3", agent_path)
    # Can't easily import due to ollama dependency at top level
    # Instead, let's exec just the list definitions

    # Extract simple list
    lines = content.split("\n")
    in_simple = False
    in_complex = False
    simple_buf = []
    complex_buf = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("simple_prompts_en = ["):
            in_simple = True
            simple_buf.append("[")
            continue
        if in_simple:
            simple_buf.append(line)
            if stripped == "]":
                in_simple = False
                continue

        if stripped.startswith("complex_prompts_en = ["):
            in_complex = True
            complex_buf.append("[")
            continue
        if in_complex:
            complex_buf.append(line)
            if stripped == "]":
                in_complex = False
                continue

    try:
        simple_prompts = eval("\n".join(simple_buf))
    except Exception:
        simple_prompts = []
    try:
        complex_prompts = eval("\n".join(complex_buf))
    except Exception:
        complex_prompts = []

    legacy = []
    pid = 1
    for text in simple_prompts:
        legacy.append({
            "prompt_id": f"leg_{pid:04d}",
            "domain": "general",
            "complexity": "simple",
            "text": text.strip(),
            "source": "legacy",
        })
        pid += 1
    for text in complex_prompts:
        legacy.append({
            "prompt_id": f"leg_{pid:04d}",
            "domain": "general",
            "complexity": "complex",
            "text": text.strip(),
            "source": "legacy",
        })
        pid += 1
    return legacy


def main():
    # Build cyber-IR prompts
    cyber_ir = _build_cyber_ir_prompts()
    cyber_ir_path = os.path.join(SCRIPT_DIR, "j3_prompt_set_cyber_ir.jsonl")
    with open(cyber_ir_path, "w", encoding="utf-8") as f:
        for p in cyber_ir:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Cyber-IR prompts: {len(cyber_ir)}")
    simple_count = sum(1 for p in cyber_ir if p["complexity"] == "simple")
    medium_count = sum(1 for p in cyber_ir if p["complexity"] == "medium")
    complex_count = sum(1 for p in cyber_ir if p["complexity"] == "complex")
    print(f"  simple={simple_count}, medium={medium_count}, complex={complex_count}")
    print(f"  Written to {cyber_ir_path}")

    # Load legacy prompts
    legacy = _load_legacy_prompts()
    print(f"\nLegacy prompts: {len(legacy)}")

    # Build combined set
    combined = []
    for p in cyber_ir:
        row = dict(p)
        row["source"] = "cyber_ir"
        combined.append(row)
    combined.extend(legacy)

    combined_path = os.path.join(SCRIPT_DIR, "j3_prompt_set_combined.jsonl")
    with open(combined_path, "w", encoding="utf-8") as f:
        for p in combined:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nCombined prompts: {len(combined)}")
    print(f"  Written to {combined_path}")


if __name__ == "__main__":
    main()
