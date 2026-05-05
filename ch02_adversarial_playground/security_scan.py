import subprocess
import json
import sys
from pathlib import Path
from typing import Optional


# Modelscan wrapper that will detect serialization attacks in static model files
def run_modelscan(model_path: Path) -> dict:
    """
    Scan a model file for serialization vulnerabiliites.
    Detects malicious code embedded in pickle, PyTorch, and Keras formats.
    """
    print(f"\nRunning ModelScan on: {model_path}")
    print("-" * 50)

    result = subprocess.run(
        ["modelscan", "--path", str(model_path), "--json"],
        capture_output=True,
        text=True
    )

    if result.returncode not in [0, 1]:
        # 0 - scan complete, no issues found
        # 1 - scan complete, issues found
        print(f"ModelScan error: {result.stderr}")
        return {"status": "error", "details": result.stderr}
    
    try:
        scan_result = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(result.stdout)
        return {"status": "complete", "details": result.stdout}
    
    issues = scan_result.get("issues", [])

    if issues:
        print(f"Issues found: {len(issues)}")
        for issue in issues:
            print(f"    - {issues}")
        else:
            print("No issues found.  Model appears safe")

        return scan_result
    

def run_bandit(code_path: Path) -> Path:
    """
    Run Bandit static analysis on Python code.
    Detects common security issues in source files.
    """
    print(f"\nRunning Bandit on: {code_path}")
    print("-" * 50)

    result = subprocess.run(
        ["bandit", "-r", str(code_path), 
         "--exclude", str(code_path / ".venv"),
         "-f", "json", "-q"],
        capture_output=True,
        text=True
    )

    try:
        scan_result = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(result.stdout)
        return {"status": "complete", "details": result.stdout}
    
    metrics = scan_result.get("metrics", {}).get("_totals", {})
    issues = scan_result.get("results", [])

    print(f"Issues found: {len(issues)}")
    print(f"High severity: {metrics.get('SEVERITY.HIGH', 0)}")
    print(f"Medium severity: {metrics.get('SEVERITY.MEDIUM', 0)}")
    print(f"Low severity: {metrics.get('SEVERITY.LOW', 0)}")

    if issues:
        print("\nTop issues:")
        for issue in issues[:5]:
            print(f" [{issue['issue_severity']}] {issue['issue_text']}")
            print(f" {issue['filename']}:{issue['line_number']}")

    return scan_result


def run_safety(requirements_path: Path) -> dict:
    """
    Scan dependencies for known vulnerabilities using Safety.
    Checks requirements.txt against Safety CVE vulnerability database.
    """
    print(f"\nRunning Safety on: {requirements_path}")
    print("-" * 50)

    result = subprocess.run(
        ["safety", "check", "-r", str(requirements_path), "--json"],
        capture_output=True,
        text=True
    )

    try:
        scan_result = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(result.stdout)
        return {"status": "complete", "details": result.stdout}
    
    vulnerabilities = scan_result.get("vulnerabilities", [])

    if vulnerabilities:
        print(f"Vulnerabilities found: {len(vulnerabilities)}")
        for vuln in vulnerabilities:
            print(f" [{vuln.get('severity', 'UNKNOWN')}] "
                  f"{vuln.get('package_name')} {vuln.get('analyzed_version')}")
            print(f" {vuln.get('advisory')[:80]}...")
        else:
            print("No vulnerabilities found.")

    return scan_result


def run_full_security_scan(
        repo_path: Path,
        model_path: Optional[Path] = None
) -> dict:
    """
    Run full security scan suite against the repo.
    ModelScan on model artifact, Bandit on source code.
    Safety on requirements.txt.
    """
    print("\n" + "-" * 50)
    print("ADVERSARIAL AI LAB - SECURITY SCAN")
    print("=" * 50)

    results = {}

    if model_path and model_path.exists():
        results["modelscan"] = run_modelscan(model_path)
    else:
        print("\nModelScan: no such model path provided or file not found, skipping.")
        results["modelscan"] = {"status": "skipped"}

    results["bandit"] = run_bandit(repo_path)

    requirements_path = repo_path / "requirements.txt"
    if requirements_path.exists():
        results["safety"] = run_safety(requirements_path)
    else:
        print("\nSafety: requirements.txt not found, skipping")

    print("\n" + "=" * 50)
    print("SCAN COMPLETE")
    print("=" * 50)

    return results


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    model_path = repo_path / "shared" / "models" / "best_checkpoint.pt"
    run_full_security_scan(repo_path, model_path)


