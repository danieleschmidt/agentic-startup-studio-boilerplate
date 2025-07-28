# Security Policy

## Supported Versions

We actively support security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. Please follow responsible disclosure practices:

### How to Report

**DO NOT** open public GitHub issues for security vulnerabilities.

Instead, please email security issues to: **security@terragon.ai**

Include the following information:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Any suggested fixes (if available)

### What to Expect

- **Initial Response**: Within 48 hours
- **Status Updates**: Every 7 days until resolved
- **Resolution Timeline**: Critical issues within 7 days, others within 30 days

### Security Features

Our template includes security best practices:

#### Container Security
- Multi-stage Docker builds to minimize attack surface
- Non-root user execution in containers
- Regular base image updates
- Vulnerability scanning in CI/CD

#### Application Security
- Input validation with Pydantic models
- SQL injection prevention via SQLAlchemy ORM
- CORS configuration for API security
- Rate limiting middleware
- JWT token authentication
- Environment variable protection

#### Infrastructure Security
- TLS encryption for all communications
- Network isolation with Docker Compose
- Secrets management via environment variables
- Regular dependency updates
- Security scanning automation

#### Development Security
- Pre-commit hooks for security checks
- Automated dependency vulnerability scanning
- SAST (Static Application Security Testing)
- License compliance checking

## Security Scanning

The template includes automated security scanning:

```bash
# Run security scan
python security_scanner.py

# Check dependencies
pip-audit --requirement requirements.txt

# Container security scan
docker scout cves your-image:tag
```

## Security Hardening Checklist

When deploying to production, ensure:

- [ ] Change all default passwords and secrets
- [ ] Enable TLS/SSL certificates
- [ ] Configure proper firewall rules
- [ ] Set up log monitoring and alerting
- [ ] Implement backup and disaster recovery
- [ ] Regular security updates and patches
- [ ] Enable container security scanning
- [ ] Configure intrusion detection
- [ ] Implement rate limiting and DDoS protection
- [ ] Set up vulnerability monitoring

## Compliance

This template supports compliance with:
- **OWASP Top 10**: Web application security risks
- **NIST Cybersecurity Framework**: Security standards
- **GDPR**: Data protection regulations (with proper configuration)
- **SOC 2**: Security controls for service organizations
- **SLSA**: Supply chain security framework

## Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [React Security](https://reactjs.org/docs/dom-elements.html#dangerouslysetinnerhtml)

## License and Legal

Security reports and discussions are covered under our standard [MIT License](LICENSE) and [Code of Conduct](CODE_OF_CONDUCT.md).