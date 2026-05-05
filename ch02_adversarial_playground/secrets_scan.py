import subprocess
import json
from pathlib import Path


def create_baseline(repo_path: Path) -> None:
    """
    Create a detect-secrets baseline file for the repo.
    Run once to establish known secrest state.
    Baseline file should be committed to the repo.
    """
    baseline_path = repo_path / ".secrets.baseline"

    print(f"\nCreating secrets baseline at: {baseline_path}")
    print("-" * 50)

    result = subprocess.run(
        ["detect-secrets", "scan", "--base64-limit", "4.5",
         "--exclude-files", r".*\.ipynb$", # Colab notebooks are known to trigger false positives
         str(repo_path)],
        capture_output=True,
        text=True,
        cwd=str(repo_path)
    )

    with open(baseline_path, "w") as f:
        f.write(result.stdout)

    baseline = json.loads(result.stdout)
    secret_count = sum(
        len(v) for v in baseline.get("results", {}).values()
    )

    print(f"Baseline created.  Potential secrets found: {secret_count}")
    print(f"Review baseline before committing - audit any flagged items.")


def run_secrets_scan(repo_path: Path) -> dict:
    """
    Scan repo against existing baseline for new secrets.
    Run after baseline is established to catch newly introduced secrets.
    """
    baseline_path = repo_path / ".secrets.baseline"

    if not baseline_path.exists():
        print("No baseline found.  Run create_baseline() first.")
        return {"status": "no_baseline"}
    
    print(f"\nRunning secrets scan against baseline...")
    print("-" * 50)

    result = subprocess.run(
        ["detect-secrets", "scan", "--base64-limit", "4.5",
         "--exclude-files", r".*\.ipynb$",
          str(repo_path)],
        capture_output=True,
        text=True,
        cwd=str(repo_path)
    )

    try:
        current = json.loads(result.stdout)
        with open(baseline_path) as f:
            baseline = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing scan output: {e}")
        return {"status": "error"}
    
    baseline_files = set(baseline.get("results", {}).keys())
    current_files = set(current.get("results", {}).keys())
    new_files = current_files - baseline_files

    new_secrets = []
    for file in current_files:
        current_secrets = current["results"].get(file, [])
        baseline_secrets = baseline["results"].get(file, [])
        baseline_types = {s["type"] for s in baseline_secrets}
        for secret in current_secrets:
            if secret["type"] not in baseline_types:
                new_secrets.append({
                    "file": file,
                    "type": secret["type"],
                    "line": secret["line_number"]
                })

    if new_secrets:
        print(f"NEW secrets detected: {len(new_secrets)}")
        for s in new_secrets:
            print(f" [{s['type']}] {s['file']}:{s['line']}")
    else:
        print("No new secrets detected since baseline.")

    return {
        "status": "complete",
        "new_secrets": new_secrets,
        "new_files_with_secrets": list(new_files)
    }


if __name__ == "__main__":
    import sys

    repo_path = Path(__file__).parent.parent

    if len(sys.argv) > 1 and sys.argv[1] == "baseline":
        create_baseline(repo_path)
    else:
        result = run_secrets_scan(repo_path)
        if result.get("new_secrets"):
            sys.exit(1)
        else:
            sys.exit(0)