# Contributing to Agentic Startup Studio Boilerplate

Thank you for your interest in contributing to this cookiecutter template! We welcome contributions that help improve the template for the agentic startup community.

## 🚀 Getting Started

### Development Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/agentic-startup-studio-boilerplate.git
   cd agentic-startup-studio-boilerplate
   ```

2. **Install development tools**
   ```bash
   pip install cookiecutter pyyaml yamllint
   ```

3. **Test the template**
   ```bash
   # Run validation tests
   python test_cookiecutter_config.py
   python test_readme_badges.py
   python test_ci_workflow.py
   
   # Test template generation
   cookiecutter . --no-input --output-dir /tmp/test-output
   ```

## 📝 How to Contribute

### Reporting Issues

- **Bug Reports**: Use the issue template and include:
  - Template generation steps that failed
  - Error messages
  - Expected vs actual behavior
  - Your environment (OS, Python version, cookiecutter version)

- **Feature Requests**: Describe:
  - The agentic startup use case
  - How it would improve the template
  - Any examples from other projects

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Follow our standards**
   - Update `cookiecutter.json` for new template variables
   - Add tests for new functionality
   - Update documentation and changelog
   - Follow existing code patterns

3. **Test your changes**
   ```bash
   # Run all validation tests
   python test_*.py
   
   # Test template generation with various configurations
   cookiecutter . --no-input
   cookiecutter . --config-file test-configs/minimal.yml
   ```

4. **Submit a pull request**
   - Clear title and description
   - Link to related issues
   - Include test results
   - Update CHANGELOG.md

### Code Standards

- **Template Files**: Follow language-specific best practices
- **Documentation**: Use clear, concise language
- **Configuration**: Provide sensible defaults
- **Testing**: Maintain high test coverage

## 🏗️ Template Structure

```
agentic-startup-studio-boilerplate/
├── cookiecutter.json           # Template configuration
├── {{cookiecutter.project_slug}}/  # Generated project files
├── hooks/                      # Pre/post generation scripts
├── tests/                      # Template validation tests
└── docs/                       # Template documentation
```

## 🤖 Agentic Startup Focus

This template specifically targets AI-powered startups. When contributing, consider:

- **AI Frameworks**: CrewAI, LangChain, AutoGen integration examples
- **Infrastructure**: Scalable deployment for AI workloads
- **Development Experience**: Tools that accelerate AI product development
- **Best Practices**: Security, monitoring, and governance for AI systems

## 🚦 Pull Request Process

1. **Before submitting**:
   - Ensure all tests pass
   - Update documentation
   - Add changelog entry
   - Squash commits into logical units

2. **Review process**:
   - Automated CI checks must pass
   - At least one maintainer review required
   - Template generation testing in CI
   - Community feedback period for major changes

3. **After merge**:
   - Update any related documentation
   - Consider if examples need updating
   - Test the published template

## 📞 Getting Help

- **Questions**: Open a discussion or issue
- **Real-time chat**: Join our community Discord
- **Documentation**: Check the README and docs/ folder

## 🎯 Priority Areas

We're especially interested in contributions for:

- [ ] More AI framework integrations (LangChain, etc.)
- [ ] Advanced deployment configurations
- [ ] Security hardening examples
- [ ] Performance optimization templates
- [ ] Testing strategies for AI applications

Thank you for contributing to the agentic startup ecosystem! 🚀