# Developer Onboarding Guide

Welcome to the Agentic Startup Studio project! This guide will help you get up and running quickly.

## Prerequisites

- **Docker** and **Docker Compose** installed
- **Node.js 18+** and **npm**
- **Python 3.11+** and **pip**
- **Git** for version control
- **VS Code** (recommended) with devcontainer support

## Quick Start

### Option 1: Using DevContainer (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd agentic-startup-studio-boilerplate
   ```

2. **Open in VS Code**:
   ```bash
   code .
   ```

3. **Reopen in DevContainer**:
   - VS Code will prompt to reopen in devcontainer
   - Or use Command Palette: "Dev Containers: Reopen in Container"

4. **Wait for setup to complete** (automated post-create script runs)

5. **Start development environment**:
   ```bash
   dev-up
   ```

### Option 2: Local Development

1. **Clone and install dependencies**:
   ```bash
   git clone <repository-url>
   cd agentic-startup-studio-boilerplate
   npm run setup
   ```

2. **Start development environment**:
   ```bash
   npm run dev:up
   ```

## Development Workflow

### Daily Development

```bash
# Start all services
npm run dev:up

# View logs
npm run dev:logs

# Run tests
npm run test

# Lint and format code
npm run lint
npm run lint:fix

# Stop services
npm run dev:down
```

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards

3. **Run quality checks**:
   ```bash
   npm run lint
   npm run typecheck
   npm run test
   ```

4. **Commit your changes** (using conventional commits):
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Project Structure

```
├── backend/                 # FastAPI backend
│   ├── main.py             # Application entry point
│   ├── api/                # API routes
│   ├── models/             # Database models
│   ├── services/           # Business logic
│   └── tests/              # Backend tests
├── frontend/               # React frontend
│   ├── src/                # Source code
│   ├── public/             # Static assets
│   └── tests/              # Frontend tests
├── docs/                   # Documentation
│   ├── adr/                # Architecture Decision Records
│   ├── guides/             # Development guides
│   └── runbooks/           # Operational runbooks
├── tests/                  # Integration and E2E tests
├── .github/                # GitHub Actions workflows
├── .devcontainer/          # DevContainer configuration
└── docker-compose*.yml    # Docker compositions
```

## Coding Standards

### Python (Backend)

- **Formatting**: Black with line length 88
- **Import sorting**: isort with black profile
- **Linting**: flake8
- **Type checking**: mypy
- **Testing**: pytest with async support

```python
# Example code structure
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class UserCreate(BaseModel):
    name: str
    email: str
    age: Optional[int] = None

@app.post("/users/")
async def create_user(user: UserCreate) -> dict:
    """Create a new user."""
    # Implementation here
    return {"message": "User created successfully"}
```

### TypeScript (Frontend)

- **Formatting**: Prettier
- **Linting**: ESLint
- **Type checking**: TypeScript strict mode
- **Testing**: Jest + React Testing Library

```typescript
// Example component structure
interface UserProps {
  name: string;
  email: string;
  onEdit?: () => void;
}

export const UserCard: React.FC<UserProps> = ({ name, email, onEdit }) => {
  return (
    <div className="user-card">
      <h3>{name}</h3>
      <p>{email}</p>
      {onEdit && <button onClick={onEdit}>Edit</button>}
    </div>
  );
};
```

### Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```bash
feat: add user authentication system
fix: resolve database connection timeout
docs: update API documentation
test: add unit tests for user service
```

## Testing

### Backend Testing

```bash
# Run all backend tests
cd backend && python -m pytest

# Run with coverage
cd backend && python -m pytest --cov=. --cov-report=html

# Run specific test file
cd backend && python -m pytest tests/test_users.py

# Run specific test
cd backend && python -m pytest tests/test_users.py::test_create_user
```

### Frontend Testing

```bash
# Run all frontend tests
cd frontend && npm test

# Run with coverage
cd frontend && npm run test:coverage

# Run specific test file
cd frontend && npm test UserCard.test.tsx

# Run E2E tests
npm run test:e2e
```

### Integration Testing

```bash
# Run integration tests
cd tests/integration && python -m pytest

# Run E2E tests with Playwright
npm run test:e2e
```

## Debugging

### Backend Debugging

1. **VS Code Launch Configuration**:
   - Use the "Debug FastAPI Backend" configuration
   - Set breakpoints in your Python code
   - Start debugging with F5

2. **Manual Debugging**:
   ```bash
   cd backend
   python -m debugpy --listen 5678 --wait-for-client main.py
   ```

### Frontend Debugging

1. **Browser DevTools**: Use React DevTools extension
2. **VS Code Debugging**: Use "Debug React Frontend" configuration
3. **Console Logging**: Use `console.log()` for quick debugging

### Database Debugging

1. **Access database directly**:
   ```bash
   docker-compose exec postgres psql -U postgres -d devdb
   ```

2. **View database in Adminer**: http://localhost:8081

## Environment Variables

### Development

Create `.env.local` files for local development:

```bash
# Backend (.env.local in backend/)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/devdb
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key-here
OPENAI_API_KEY=your-openai-key-here

# Frontend (.env.local in frontend/)
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ENVIRONMENT=development
```

### Production

Use environment-specific configurations and secrets management.

## Common Issues and Solutions

### Port Conflicts

If you get port conflict errors:
```bash
# Check what's using the port
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use different ports
export API_PORT=8001
export FRONTEND_PORT=3001
```

### Database Connection Issues

```bash
# Reset database
docker-compose down -v
docker-compose up -d postgres
npm run db:migrate
```

### Node Modules Issues

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

## Getting Help

1. **Documentation**: Check the `/docs` directory
2. **Issues**: Search existing GitHub issues
3. **Discussions**: Use GitHub Discussions for questions
4. **Code Review**: Create draft PRs for early feedback

## Contributing

1. Read the [CONTRIBUTING.md](../CONTRIBUTING.md) guide
2. Follow the coding standards
3. Write tests for new features
4. Update documentation as needed
5. Ensure all CI checks pass

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [Docker Documentation](https://docs.docker.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

Happy coding! 🚀