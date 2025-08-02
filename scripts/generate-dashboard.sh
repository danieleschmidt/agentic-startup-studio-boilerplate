#!/bin/bash

# Generate Project Health Dashboard
# This script runs all dashboard generation tools and creates a comprehensive project status report

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_DIR="./dashboard-output"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo -e "${BLUE}🚀 Generating Project Health Dashboard${NC}"
echo "======================================"

# Create output directory
echo -e "${YELLOW}📁 Creating output directory...${NC}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/reports"
mkdir -p "$OUTPUT_DIR/charts"
mkdir -p "$OUTPUT_DIR/metrics"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to run command with error handling
run_command() {
    local cmd="$1"
    local description="$2"
    
    echo -e "${YELLOW}⚡ $description...${NC}"
    
    if eval "$cmd"; then
        echo -e "${GREEN}✅ $description completed${NC}"
    else
        echo -e "${RED}❌ $description failed${NC}"
        return 1
    fi
}

# Collect project metrics
echo -e "\n${BLUE}📊 Collecting Project Metrics${NC}"
echo "================================"

if command_exists python3; then
    run_command "python3 scripts/automation/metrics_collector.py" "Collecting comprehensive metrics"
else
    echo -e "${RED}❌ Python3 not found, skipping metrics collection${NC}"
fi

# Generate health dashboard
echo -e "\n${BLUE}🏥 Generating Health Dashboard${NC}"
echo "================================"

if command_exists python3; then
    run_command "python3 scripts/automation/health_dashboard.py --output $OUTPUT_DIR" "Generating health dashboard"
else
    echo -e "${RED}❌ Python3 not found, skipping health dashboard${NC}"
fi

# Run repository maintenance check
echo -e "\n${BLUE}🔧 Running Repository Maintenance${NC}"
echo "=================================="

if command_exists python3; then
    run_command "python3 scripts/automation/repo_maintenance.py --dry-run --report $OUTPUT_DIR/reports/maintenance-report.json" "Repository maintenance analysis"
else
    echo -e "${RED}❌ Python3 not found, skipping maintenance check${NC}"
fi

# Security analysis
echo -e "\n${BLUE}🔒 Running Security Analysis${NC}"
echo "============================="

# Bandit security scan
if command_exists bandit; then
    run_command "bandit -r . -f json -o $OUTPUT_DIR/reports/security-scan.json" "Python security scan"
else
    echo -e "${YELLOW}⚠️  Bandit not found, skipping Python security scan${NC}"
fi

# Safety dependency check
if command_exists safety; then
    run_command "safety check --json --output $OUTPUT_DIR/reports/dependency-vulnerabilities.json" "Dependency vulnerability scan"
else
    echo -e "${YELLOW}⚠️  Safety not found, skipping dependency scan${NC}"
fi

# NPM audit (if package.json exists)
if [[ -f "package.json" ]] && command_exists npm; then
    run_command "npm audit --json > $OUTPUT_DIR/reports/npm-audit.json" "NPM dependency audit"
else
    echo -e "${YELLOW}⚠️  NPM or package.json not found, skipping NPM audit${NC}"
fi

# Trivy container scan (if Dockerfile exists)
if [[ -f "Dockerfile" ]] && command_exists trivy; then
    run_command "trivy fs --format json --output $OUTPUT_DIR/reports/container-vulnerabilities.json ." "Container vulnerability scan"
else
    echo -e "${YELLOW}⚠️  Trivy or Dockerfile not found, skipping container scan${NC}"
fi

# Code quality analysis
echo -e "\n${BLUE}🎯 Analyzing Code Quality${NC}"
echo "=========================="

# Flake8 linting
if command_exists flake8; then
    run_command "flake8 --statistics --count --format=json --output-file=$OUTPUT_DIR/reports/flake8-report.json" "Python code linting"
else
    echo -e "${YELLOW}⚠️  Flake8 not found, skipping Python linting${NC}"
fi

# Mypy type checking
if command_exists mypy; then
    run_command "mypy . --json-report $OUTPUT_DIR/reports/mypy-report" "Python type checking"
else
    echo -e "${YELLOW}⚠️  Mypy not found, skipping type checking${NC}"
fi

# Test coverage
echo -e "\n${BLUE}🧪 Analyzing Test Coverage${NC}"
echo "=========================="

if command_exists pytest; then
    run_command "pytest --cov=. --cov-report=json:$OUTPUT_DIR/reports/coverage.json --cov-report=html:$OUTPUT_DIR/reports/coverage-html" "Test coverage analysis"
else
    echo -e "${YELLOW}⚠️  Pytest not found, skipping coverage analysis${NC}"
fi

# Performance analysis
echo -e "\n${BLUE}⚡ Performance Analysis${NC}"
echo "======================="

if command_exists python3; then
    run_command "python3 scripts/automation/performance_analyzer.py --output $OUTPUT_DIR/reports/performance-report.json" "Performance analysis"
else
    echo -e "${YELLOW}⚠️  Performance analyzer not available${NC}"
fi

# Generate documentation metrics
echo -e "\n${BLUE}📚 Documentation Analysis${NC}"
echo "========================="

# Count documentation files
DOC_FILES=$(find . -name "*.md" -not -path "./node_modules/*" -not -path "./.git/*" | wc -l)
README_EXISTS="false"
[[ -f "README.md" ]] && README_EXISTS="true"
CHANGELOG_EXISTS="false"
[[ -f "CHANGELOG.md" ]] && CHANGELOG_EXISTS="true"
CONTRIBUTING_EXISTS="false"
[[ -f "CONTRIBUTING.md" ]] && CONTRIBUTING_EXISTS="true"

# Create documentation report
cat > "$OUTPUT_DIR/reports/documentation-metrics.json" << EOF
{
  "total_markdown_files": $DOC_FILES,
  "readme_exists": $README_EXISTS,
  "changelog_exists": $CHANGELOG_EXISTS,
  "contributing_exists": $CONTRIBUTING_EXISTS,
  "docs_directory_exists": $([ -d "docs" ] && echo "true" || echo "false"),
  "last_updated": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

echo -e "${GREEN}✅ Documentation analysis completed${NC}"

# Git repository analysis
echo -e "\n${BLUE}📝 Git Repository Analysis${NC}"
echo "=========================="

# Git metrics
TOTAL_COMMITS=$(git rev-list --all --count 2>/dev/null || echo "0")
TOTAL_BRANCHES=$(git branch -r | wc -l 2>/dev/null || echo "0")
CONTRIBUTORS=$(git shortlog -sn | wc -l 2>/dev/null || echo "0")
LAST_COMMIT_DATE=$(git log -1 --format="%ci" 2>/dev/null || echo "unknown")

# Create git report
cat > "$OUTPUT_DIR/reports/git-metrics.json" << EOF
{
  "total_commits": $TOTAL_COMMITS,
  "total_branches": $TOTAL_BRANCHES,
  "contributors": $CONTRIBUTORS,
  "last_commit_date": "$LAST_COMMIT_DATE",
  "repository_age_days": $(( ($(date +%s) - $(git log --reverse --format="%at" | head -1 2>/dev/null || echo $(date +%s))) / 86400 )),
  "last_updated": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

echo -e "${GREEN}✅ Git analysis completed${NC}"

# Generate summary report
echo -e "\n${BLUE}📋 Generating Summary Report${NC}"
echo "============================="

# Create comprehensive summary
cat > "$OUTPUT_DIR/dashboard-summary.md" << EOF
# Project Health Dashboard Summary

**Generated:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Repository:** $(git remote get-url origin 2>/dev/null || echo "Local repository")

## 📊 Key Metrics

### Git Repository
- **Total Commits:** $TOTAL_COMMITS
- **Contributors:** $CONTRIBUTORS
- **Branches:** $TOTAL_BRANCHES
- **Last Commit:** $LAST_COMMIT_DATE

### Documentation
- **Markdown Files:** $DOC_FILES
- **README:** $README_EXISTS
- **CHANGELOG:** $CHANGELOG_EXISTS
- **Contributing Guide:** $CONTRIBUTING_EXISTS

### Security & Quality
- Security scans completed: $(ls $OUTPUT_DIR/reports/*security* 2>/dev/null | wc -l)
- Quality reports generated: $(ls $OUTPUT_DIR/reports/*report* 2>/dev/null | wc -l)

## 📁 Generated Reports

### Health Dashboard
- \`health-dashboard.html\` - Interactive HTML dashboard
- \`health-report.json\` - Machine-readable health metrics

### Security Reports
$(find $OUTPUT_DIR/reports -name "*security*" -o -name "*vulnerabilities*" -o -name "*audit*" | sed 's/^/- /')

### Quality Reports
$(find $OUTPUT_DIR/reports -name "*report*" -o -name "*coverage*" -o -name "*metrics*" | sed 's/^/- /')

### Charts and Visualizations
$(find $OUTPUT_DIR/charts -name "*.png" 2>/dev/null | sed 's/^/- /' || echo "- No charts generated")

## 🎯 Next Steps

1. Review the health dashboard: \`$OUTPUT_DIR/health-dashboard.html\`
2. Check security reports for vulnerabilities
3. Examine code quality metrics
4. Implement recommendations from maintenance report

## 🔗 Quick Links

- [Health Dashboard]($OUTPUT_DIR/health-dashboard.html)
- [Project Metrics](.github/project-metrics.json)
- [Security Reports]($OUTPUT_DIR/reports/)

---
*This report was generated automatically by the Agentic Startup Studio dashboard generator.*
EOF

# Create index.html for easy navigation
cat > "$OUTPUT_DIR/index.html" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Dashboard - Agentic Startup Studio</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f8fafc; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        h1 { color: #1e293b; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; }
        .link-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }
        .link-card { padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; text-decoration: none; color: #1e293b; transition: all 0.2s; }
        .link-card:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        .link-title { font-weight: 600; margin-bottom: 8px; color: #3b82f6; }
        .link-desc { font-size: 0.9em; color: #64748b; }
        .timestamp { color: #64748b; font-size: 0.9em; margin-top: 30px; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Project Dashboard</h1>
        <p>Welcome to the Agentic Startup Studio project dashboard. Explore the reports and metrics below:</p>
        
        <div class="link-grid">
            <a href="health-dashboard.html" class="link-card">
                <div class="link-title">🏥 Health Dashboard</div>
                <div class="link-desc">Interactive health metrics and recommendations</div>
            </a>
            
            <a href="reports/" class="link-card">
                <div class="link-title">📊 Reports</div>
                <div class="link-desc">Detailed analysis reports (security, quality, performance)</div>
            </a>
            
            <a href="charts/" class="link-card">
                <div class="link-title">📈 Charts</div>
                <div class="link-desc">Visual metrics and trend analysis</div>
            </a>
            
            <a href="dashboard-summary.md" class="link-card">
                <div class="link-title">📋 Summary</div>
                <div class="link-desc">Complete dashboard summary in Markdown</div>
            </a>
        </div>
        
        <div class="timestamp">
            Generated on $(date -u +"%Y-%m-%d at %H:%M:%S UTC")
        </div>
    </div>
</body>
</html>
EOF

# Final summary
echo -e "\n${GREEN}🎉 Dashboard Generation Complete!${NC}"
echo "================================="
echo -e "📁 Output directory: ${BLUE}$OUTPUT_DIR${NC}"
echo -e "🌐 Open dashboard: ${BLUE}$OUTPUT_DIR/index.html${NC}"
echo -e "📊 Health dashboard: ${BLUE}$OUTPUT_DIR/health-dashboard.html${NC}"
echo -e "📋 Summary report: ${BLUE}$OUTPUT_DIR/dashboard-summary.md${NC}"
echo ""
echo -e "${YELLOW}💡 Tip: Open the HTML files in your browser for interactive reports${NC}"

# Archive previous dashboards (optional)
if [[ "${ARCHIVE_PREVIOUS:-false}" == "true" ]]; then
    ARCHIVE_DIR="./dashboard-archive/$TIMESTAMP"
    mkdir -p "$ARCHIVE_DIR"
    cp -r "$OUTPUT_DIR"/* "$ARCHIVE_DIR/" 2>/dev/null || true
    echo -e "${GREEN}📦 Previous dashboard archived to: $ARCHIVE_DIR${NC}"
fi

echo -e "\n${GREEN}✅ All done! Happy coding! 🚀${NC}"