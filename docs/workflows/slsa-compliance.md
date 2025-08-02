# SLSA Compliance Guide

## Overview

This guide outlines how the Agentic Startup Studio Boilerplate implements SLSA (Supply-chain Levels for Software Artifacts) compliance to ensure software supply chain security.

## SLSA Framework

SLSA is a security framework for ensuring the integrity of software artifacts throughout the software supply chain. It consists of four levels:

- **SLSA 1**: Basic build system integrity
- **SLSA 2**: Authenticated build platforms and signed provenance
- **SLSA 3**: Hardened builds and stronger protections
- **SLSA 4**: Highest level with hermetic builds and formal verification

## Current Implementation: SLSA Level 2

### Build Platform Requirements ✅

#### Authenticated Build Platform
- **GitHub Actions**: Our CI/CD runs on GitHub-hosted runners
- **Identity**: Build platform identity is authenticated via GitHub's OIDC tokens
- **Isolation**: Each build runs in isolated environments

#### Signed Provenance
```yaml
# .github/workflows/slsa-provenance.yml
name: SLSA Provenance
on:
  release:
    types: [published]

jobs:
  provenance:
    permissions:
      actions: read
      id-token: write
      contents: write
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.9.0
    with:
      base64-subjects: "${{ needs.build.outputs.hashes }}"
      upload-assets: true
```

#### Build Process Integrity
```yaml
# Example build step with provenance
- name: Build and Generate Provenance
  id: build
  run: |
    # Build the application
    docker build -t ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ env.TAG }} .
    
    # Generate SHA256 hash
    DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ env.TAG }} | cut -d'@' -f2)
    echo "digest=$DIGEST" >> $GITHUB_OUTPUT
    
    # Create attestation
    echo "Creating SLSA attestation for $DIGEST"
```

### Security Controls

#### Source Code Protection
- **Branch Protection**: Main branch requires PR reviews and status checks
- **Signed Commits**: Encouraged for sensitive changes
- **Access Control**: Repository access limited to authorized personnel

```yaml
# Branch protection configuration
branch_protection_rules:
  main:
    required_status_checks:
      strict: true
      contexts:
        - ci/tests
        - ci/security-scan
        - ci/lint
    required_pull_request_reviews:
      required_approving_review_count: 2
      dismiss_stale_reviews: true
      require_code_owner_reviews: true
    restrictions:
      users: []
      teams: ["core-team"]
    enforce_admins: true
```

#### Build Environment Security
- **Immutable Runners**: GitHub-hosted runners provide clean environments
- **Dependency Pinning**: All dependencies use specific versions or hashes
- **Secrets Management**: Secure handling of sensitive build-time data

```yaml
# Example of pinned dependencies in CI
- name: Setup Node.js
  uses: actions/setup-node@b39b52d1213e96004bfcb1c61a8a6fa8ab84f3e8 # v4.0.1
  with:
    node-version: '18'
    
- name: Setup Python
  uses: actions/setup-python@0a5c61591373683505ea898e09a3ea4f39ef2b9c # v5.0.0
  with:
    python-version: '3.11'
```

#### Provenance Generation
```yaml
# Provenance generation for container images
- name: Generate SBOM
  uses: anchore/sbom-action@78fc58e266e87a38d4194b2137a3d4e9bcaf7ca1 # v0.14.3
  with:
    image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ env.TAG }}
    artifact-name: sbom.spdx.json
    output-file: ./sbom.spdx.json

- name: Sign Container Image
  run: |
    cosign sign --yes ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ env.TAG }}

- name: Generate Provenance
  uses: slsa-framework/slsa-github-generator/actions/generator/container@v1.9.0
  with:
    image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
    digest: ${{ steps.build.outputs.digest }}
    registry-username: ${{ github.actor }}
    registry-password: ${{ secrets.GITHUB_TOKEN }}
```

## Verification Procedures

### Verifying Provenance
```bash
# Install slsa-verifier
go install github.com/slsa-framework/slsa-verifier/v2/cli/slsa-verifier@latest

# Verify container image provenance
slsa-verifier verify-image ghcr.io/danieleschmidt/agentic-startup-studio:latest \
  --source-uri github.com/danieleschmidt/agentic-startup-studio-boilerplate \
  --source-tag v1.0.0

# Verify artifact provenance
slsa-verifier verify-artifact ./binary \
  --provenance ./binary.intoto.jsonl \
  --source-uri github.com/danieleschmidt/agentic-startup-studio-boilerplate
```

### SBOM Verification
```bash
# Verify SBOM with Syft
syft scan ghcr.io/danieleschmidt/agentic-startup-studio:latest -o spdx-json

# Validate SBOM against known vulnerabilities
grype sbom:./sbom.spdx.json
```

### Digital Signature Verification
```bash
# Verify container signature with Cosign
cosign verify ghcr.io/danieleschmidt/agentic-startup-studio:latest \
  --certificate-identity-regexp 'https://github\.com/danieleschmidt/agentic-startup-studio-boilerplate/\.github/workflows/.+' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

## SLSA Level 3 Roadmap

### Enhanced Build Platform Security
- [ ] **Hermetic Builds**: Implement hermetic build environments
- [ ] **Build Parameter Verification**: Verify all build parameters
- [ ] **Dependency Isolation**: Isolate build-time dependencies

### Advanced Provenance
- [ ] **Multi-Step Provenance**: Track provenance across multiple build steps
- [ ] **Dependency Provenance**: Include dependency provenance information
- [ ] **Reproducible Builds**: Implement reproducible build processes

```yaml
# Example hermetic build configuration
- name: Hermetic Build
  uses: docker://gcr.io/distroless/base-debian11:debug
  with:
    entrypoint: /bin/sh
    args: |
      -c "
      # Hermetic build script
      set -euo pipefail
      
      # Verify environment isolation
      [ -z \"$HOME\" ] || exit 1
      [ $(id -u) -eq 65532 ] || exit 1
      
      # Build with verified dependencies
      ./build.sh --hermetic
      "
```

### Supply Chain Monitoring
- [ ] **Dependency Monitoring**: Continuous monitoring of dependencies
- [ ] **Vulnerability Tracking**: Automated vulnerability assessment
- [ ] **Policy Enforcement**: Automated policy enforcement

## Security Policies

### Dependency Management Policy
```yaml
# dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "security-team"
    assignees:
      - "lead-developer"
    commit-message:
      prefix: "security"
      include: "scope"
    
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "security-team"
```

### Build Security Policy
```yaml
# Security policy for builds
security_requirements:
  - name: "No network access during build"
    enforcement: "strict"
    exceptions: ["dependency_download_phase"]
    
  - name: "Cryptographic verification of dependencies"
    enforcement: "strict"
    tools: ["npm_audit", "pip_audit", "trivy"]
    
  - name: "SBOM generation"
    enforcement: "required"
    format: ["spdx", "cyclonedx"]
    
  - name: "Container signing"
    enforcement: "required"
    tool: "cosign"
    key_source: "github_oidc"
```

## Compliance Monitoring

### Automated Compliance Checks
```python
# compliance_checker.py
import json
import requests
from typing import Dict, List

class SLSAComplianceChecker:
    def __init__(self, repo: str, token: str):
        self.repo = repo
        self.token = token
        self.headers = {"Authorization": f"token {token}"}
    
    def check_branch_protection(self) -> Dict:
        """Check if main branch has required protections."""
        url = f"https://api.github.com/repos/{self.repo}/branches/main/protection"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            protection = response.json()
            return {
                "protected": True,
                "required_status_checks": protection.get("required_status_checks", {}).get("strict", False),
                "required_reviews": protection.get("required_pull_request_reviews", {}).get("required_approving_review_count", 0) >= 1,
                "enforce_admins": protection.get("enforce_admins", {}).get("enabled", False)
            }
        return {"protected": False}
    
    def check_workflow_permissions(self) -> Dict:
        """Check if workflows have appropriate permissions."""
        # Implementation for checking workflow permissions
        pass
    
    def verify_signed_commits(self, commit_sha: str) -> bool:
        """Verify if commits are signed."""
        url = f"https://api.github.com/repos/{self.repo}/commits/{commit_sha}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            commit = response.json()
            return commit.get("commit", {}).get("verification", {}).get("verified", False)
        return False

# Usage in CI/CD
def run_compliance_check():
    checker = SLSAComplianceChecker("danieleschmidt/agentic-startup-studio-boilerplate", os.environ["GITHUB_TOKEN"])
    
    results = {
        "branch_protection": checker.check_branch_protection(),
        "workflow_permissions": checker.check_workflow_permissions(),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Fail if compliance requirements not met
    if not all([
        results["branch_protection"]["protected"],
        results["branch_protection"]["required_reviews"],
        results["branch_protection"]["enforce_admins"]
    ]):
        print("❌ SLSA compliance check failed")
        sys.exit(1)
    
    print("✅ SLSA compliance check passed")
    return results
```

### Compliance Reporting
```yaml
# .github/workflows/compliance-report.yml
name: SLSA Compliance Report
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:

jobs:
  compliance-report:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Compliance Check
      run: |
        python scripts/compliance_checker.py > compliance-report.json
    
    - name: Upload Compliance Report
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: compliance-report.sarif
    
    - name: Create Issue on Failure
      if: failure()
      uses: actions/github-script@v7
      with:
        script: |
          github.rest.issues.create({
            owner: context.repo.owner,
            repo: context.repo.repo,
            title: 'SLSA Compliance Check Failed',
            body: 'Automated SLSA compliance check has failed. Please review the compliance report.',
            labels: ['security', 'compliance']
          })
```

## Best Practices

### 1. Build Reproducibility
```dockerfile
# Use specific base image versions
FROM python:3.11.7-slim@sha256:2f0189b62ce8af82ce01edf9f43ac6e5de7e5967adc2d14d7c35f93ca6e22ba8

# Pin package versions
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Use specific timestamps for reproducibility
ENV SOURCE_DATE_EPOCH=1640995200
```

### 2. Dependency Verification
```bash
# Verify package integrity
pip install --require-hashes -r requirements.txt

# requirements.txt with hashes
certifi==2023.11.17 \
    --hash=sha256:9b469f3a900bf28dc19b8cfbf8019bf47f7fdd1a65a1d4ffb98fc14166beb4d1
charset-normalizer==3.3.2 \
    --hash=sha256:25baf083bf6f6b341f4121c2f3c548875ee6f5339300e08be5d0cae13fe13b7
```

### 3. Continuous Monitoring
```python
# Security monitoring script
def monitor_supply_chain():
    """Monitor supply chain security continuously."""
    
    # Check for new vulnerabilities
    vulnerabilities = scan_dependencies()
    
    # Verify artifact signatures
    verify_signatures()
    
    # Check policy compliance
    check_policies()
    
    # Generate alerts if needed
    if vulnerabilities:
        send_security_alert(vulnerabilities)
```

### 4. Documentation and Training
- Maintain up-to-date SLSA compliance documentation
- Train team members on supply chain security
- Regular security reviews and audits
- Incident response procedures for supply chain attacks

## Verification Commands

### Quick Compliance Check
```bash
# Run quick compliance verification
./scripts/verify-slsa-compliance.sh

# Check specific artifact
./scripts/verify-artifact.sh ./dist/app.tar.gz

# Verify container image
./scripts/verify-container.sh ghcr.io/danieleschmidt/agentic-startup-studio:latest
```

### Detailed Analysis
```bash
# Generate comprehensive SLSA report
python scripts/generate-slsa-report.py --output slsa-report.json

# Validate against SLSA requirements
slsa-verifier verify-provenance ./artifacts/provenance.intoto.jsonl \
  --source-uri github.com/danieleschmidt/agentic-startup-studio-boilerplate \
  --source-tag v1.0.0
```

## Resources

- [SLSA Framework Documentation](https://slsa.dev/)
- [GitHub Actions SLSA Generator](https://github.com/slsa-framework/slsa-github-generator)
- [Cosign Documentation](https://docs.sigstore.dev/cosign/overview/)
- [Supply Chain Security Best Practices](https://security.googleblog.com/2021/06/introducing-slsa-end-to-end-framework.html)

## Contact

For questions about SLSA compliance implementation:
- Security Team: security@terragon.dev
- DevOps Team: devops@terragon.dev
- Documentation: docs@terragon.dev