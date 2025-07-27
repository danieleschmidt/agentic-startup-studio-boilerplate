#!/bin/bash
set -e

echo "🚀 Starting DevContainer post-create setup..."

# Install Python dependencies if requirements exist
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
fi

if [ -f "requirements-dev.txt" ]; then
    echo "📦 Installing Python dev dependencies..."
    pip install -r requirements-dev.txt
fi

# Install pre-commit hooks if .pre-commit-config.yaml exists
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🔧 Installing pre-commit hooks..."
    pre-commit install
fi

# Install Node.js dependencies for template validation
echo "📦 Installing Node.js dependencies for validation..."
npm install -g cookiecutter-validator @commitlint/cli @commitlint/config-conventional

# Setup Git configuration if not exists
if [ ! -f ~/.gitconfig ]; then
    echo "🔧 Setting up Git configuration..."
    git config --global init.defaultBranch main
    git config --global core.autocrlf input
    git config --global pull.rebase false
fi

# Create useful aliases
echo "🔧 Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# Development aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'

# Docker aliases
alias dc='docker-compose'
alias dcu='docker-compose up'
alias dcd='docker-compose down'
alias dcl='docker-compose logs'

# Python aliases
alias py='python'
alias pip-upgrade='pip list --outdated --format=freeze | grep -v "^\-e" | cut -d = -f 1 | xargs -n1 pip install -U'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline'
alias gb='git branch'
alias gco='git checkout'

# Development shortcuts
alias dev-up='docker-compose -f docker-compose.dev.yml up -d'
alias dev-down='docker-compose -f docker-compose.dev.yml down'
alias dev-logs='docker-compose -f docker-compose.dev.yml logs -f'
alias dev-test='python -m pytest'
alias dev-lint='flake8 . && black --check . && isort --check-only .'
alias dev-format='black . && isort .'

EOF

# Install common development tools
echo "🛠️  Installing additional development tools..."
pip install --user \
    black \
    flake8 \
    isort \
    pytest \
    pytest-cov \
    pytest-asyncio \
    pre-commit \
    mypy \
    bandit \
    safety

# Create useful directories
mkdir -p .vscode/snippets
mkdir -p docs/guides
mkdir -p tests/{unit,integration,e2e}
mkdir -p scripts

# Set up VS Code snippets for common patterns
cat > .vscode/snippets/python.json << 'EOF'
{
  "FastAPI Endpoint": {
    "prefix": "fastapi-endpoint",
    "body": [
      "@app.${1:get}(\"/${2:path}\")",
      "async def ${3:function_name}(${4:params}):",
      "    \"\"\"${5:Description}\"\"\"",
      "    ${6:# Implementation}",
      "    return {\"message\": \"${7:response}\"}"
    ],
    "description": "FastAPI endpoint template"
  },
  "CrewAI Agent": {
    "prefix": "crewai-agent",
    "body": [
      "from crewai import Agent",
      "",
      "${1:agent_name} = Agent(",
      "    role=\"${2:role}\",",
      "    goal=\"${3:goal}\",",
      "    backstory=\"${4:backstory}\",",
      "    verbose=True,",
      "    allow_delegation=False",
      ")"
    ],
    "description": "CrewAI agent template"
  },
  "Pytest Test": {
    "prefix": "pytest-test",
    "body": [
      "def test_${1:function_name}():",
      "    \"\"\"Test ${2:description}.\"\"\"",
      "    # Arrange",
      "    ${3:# Setup test data}",
      "    ",
      "    # Act",
      "    ${4:# Execute function}",
      "    ",
      "    # Assert",
      "    ${5:# Verify results}"
    ],
    "description": "Pytest test template"
  }
}
EOF

echo "✅ DevContainer setup completed successfully!"
echo ""
echo "🎯 Quick Start Commands:"
echo "  dev-up      - Start development environment"
echo "  dev-test    - Run tests"
echo "  dev-lint    - Check code quality"
echo "  dev-format  - Format code"
echo ""
echo "📚 Next Steps:"
echo "  1. Run 'dev-up' to start the development environment"
echo "  2. Visit http://localhost:3000 for the frontend"
echo "  3. Visit http://localhost:8000/docs for API documentation"
echo ""