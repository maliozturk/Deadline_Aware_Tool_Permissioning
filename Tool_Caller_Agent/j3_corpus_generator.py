# =============================================================================
#  FIRM-DEADLINE TOOL CONTROL (FTC)
#  Product Signature: FTC
# ------------------------------------------------------------------------------
#  File: Tool_Caller_Agent/j3_corpus_generator.py
#  Purpose: Generate the 50-doc mock SOC corpus using Ollama.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-05-17
#  Environment: Python 3.11
# =============================================================================

"""Generate 50 mock SOC documents using a local Ollama backend.

Usage:
    python Tool_Caller_Agent/j3_corpus_generator.py
"""

import json
import os
import time

import ollama

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "j3_corpus")
GENERATION_MODEL = "qwen2.5:7b-instruct"

# Document specs: (doc_index_start, count, category, generation_prompt_template)
DOC_SPECS = [
    # 20 runbook excerpts
    (1, 4, "runbook", "Write a SOC runbook excerpt (300-500 words) for: Containment procedure for a compromised Windows endpoint detected via EDR alert. Include step-by-step actions, escalation criteria, and evidence preservation notes. Use professional incident response language."),
    (5, 4, "runbook", "Write a SOC runbook excerpt (300-500 words) for: Triage workflow for a phishing email reported by an employee. Include header analysis steps, attachment sandboxing procedure, and user communication template. Use professional SOC language."),
    (9, 4, "runbook", "Write a SOC runbook excerpt (300-500 words) for: Responding to a ransomware infection detected on a file server. Include isolation steps, backup verification, negotiation policy reference, and recovery procedure outline."),
    (13, 4, "runbook", "Write a SOC runbook excerpt (300-500 words) for: Investigating unauthorized lateral movement detected by network monitoring. Include credential reset procedures, affected host enumeration, and forensic image collection."),
    (17, 4, "runbook", "Write a SOC runbook excerpt (300-500 words) for: Handling a DDoS attack against public-facing web infrastructure. Include traffic analysis, upstream provider coordination, and WAF rule deployment steps."),
    # 15 IOC entries
    (21, 3, "ioc", "Write an IOC intelligence entry (300-500 words) about a malicious IP address 198.51.100.42 associated with command-and-control traffic. Include first-seen date, associated malware family, WHOIS summary, and recommended firewall rules. Use professional threat intel format."),
    (24, 3, "ioc", "Write an IOC intelligence entry (300-500 words) about a file hash (SHA-256: a1b2c3d4e5f6...) linked to the Cobalt Strike beacon payload. Include detection signatures, behavioral indicators, and MITRE ATT&CK mapping."),
    (27, 3, "ioc", "Write an IOC intelligence entry (300-500 words) about the domain evil-update.example.net used for DNS tunneling exfiltration. Include DNS query pattern analysis, associated campaigns, and detection rules."),
    (30, 3, "ioc", "Write an IOC intelligence entry (300-500 words) about a suspicious SSL certificate fingerprint used in man-in-the-middle attacks targeting corporate VPN users. Include certificate details and remediation steps."),
    (33, 3, "ioc", "Write an IOC intelligence entry (300-500 words) about a weaponized Office document (CVE-2024-XXXX) distributed via spear-phishing. Include macro analysis summary, payload delivery chain, and YARA rule."),
    # 15 threat-intel notes
    (36, 3, "threat_intel", "Write a threat intelligence note (300-500 words) about APT group 'Midnight Typhoon' targeting financial institutions in Southeast Asia. Include TTPs, infrastructure, and recommended defensive measures."),
    (39, 3, "threat_intel", "Write a threat intelligence note (300-500 words) about a ransomware-as-a-service campaign called 'LockByte' that targets healthcare organizations. Include affiliate model, encryption methods, and negotiation patterns."),
    (42, 3, "threat_intel", "Write a threat intelligence note (300-500 words) about a supply-chain compromise affecting a popular open-source npm package. Include timeline, affected versions, and indicators of compromise."),
    (45, 3, "threat_intel", "Write a threat intelligence note (300-500 words) about credential harvesting campaigns using fake Microsoft 365 login pages. Include infrastructure analysis, evasion techniques, and detection strategies."),
    (48, 3, "threat_intel", "Write a threat intelligence note (300-500 words) about an insider threat case involving data exfiltration through personal cloud storage. Include behavioral indicators, DLP bypass methods, and investigation approach."),
]


def generate_corpus():
    os.makedirs(CORPUS_DIR, exist_ok=True)
    generation_log = []

    for start_idx, count, category, prompt_template in DOC_SPECS:
        for i in range(count):
            doc_idx = start_idx + i
            filename = f"doc_{doc_idx:03d}.txt"
            filepath = os.path.join(CORPUS_DIR, filename)

            if os.path.exists(filepath):
                print(f"  [SKIP] {filename} already exists")
                generation_log.append({
                    "filename": filename,
                    "category": category,
                    "prompt": prompt_template,
                    "status": "skipped",
                })
                continue

            # Vary the prompt slightly per doc to get distinct content
            variation = f" Variation {i+1} of {count}: use different specific details, hostnames, IP addresses, and dates than other documents in this set."
            full_prompt = prompt_template + variation

            print(f"  Generating {filename} ({category})...", end=" ", flush=True)
            t0 = time.time()
            try:
                resp = ollama.chat(
                    model=GENERATION_MODEL,
                    messages=[{"role": "user", "content": full_prompt}],
                    options={"temperature": 0.7, "num_predict": 800, "seed": 42 + doc_idx},
                )
                content = resp.get("message", {}).get("content", "")
                elapsed = time.time() - t0
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"OK ({elapsed:.1f}s, {len(content)} chars)")
                generation_log.append({
                    "filename": filename,
                    "category": category,
                    "prompt": full_prompt,
                    "status": "generated",
                    "generation_model": GENERATION_MODEL,
                    "elapsed_sec": round(elapsed, 2),
                    "content_length_chars": len(content),
                })
            except Exception as e:
                elapsed = time.time() - t0
                print(f"ERROR ({elapsed:.1f}s): {e}")
                generation_log.append({
                    "filename": filename,
                    "category": category,
                    "prompt": full_prompt,
                    "status": "error",
                    "error": str(e),
                })

    log_path = os.path.join(CORPUS_DIR, "_generation_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(generation_log, f, indent=2, ensure_ascii=False)
    print(f"\nGeneration log written to {log_path}")
    print(f"Total docs: {len(generation_log)}")


if __name__ == "__main__":
    generate_corpus()
