# Multi-stage Dockerfile for {{cookiecutter.project_name}} Frontend
# Optimized for development and production builds

# ============================================================================
# Base Node.js image
# ============================================================================
FROM node:{{cookiecutter.node_version}}-alpine as base

# Set environment variables
ENV NODE_ENV=development
ENV NPM_CONFIG_LOGLEVEL=warn
ENV NPM_CONFIG_COLOR=false

# Install system dependencies
RUN apk add --no-cache \
    git \
    curl \
    bash

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# ============================================================================
# Development dependencies stage
# ============================================================================
FROM base as deps

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install all dependencies (including dev dependencies)
RUN npm ci --only=production --frozen-lockfile \
    && npm ci --frozen-lockfile

# ============================================================================
# Development stage
# ============================================================================
FROM base as development

WORKDIR /app

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy source code
COPY . .

# Change ownership to nodejs user
RUN chown -R nodejs:nodejs /app

# Switch to non-root user
USER nodejs

# Expose port
EXPOSE 3000

# Development command with hot reload
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# ============================================================================
# Build stage for production
# ============================================================================
FROM base as builder

WORKDIR /app

# Copy dependencies
COPY --from=deps /app/node_modules ./node_modules

# Copy source code
COPY . .

# Set production environment
ENV NODE_ENV=production
ENV GENERATE_SOURCEMAP=false

# Build the application
RUN npm run build

# ============================================================================
# Production dependencies stage
# ============================================================================
FROM base as prod-deps

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install only production dependencies
RUN npm ci --only=production --frozen-lockfile && \
    npm cache clean --force

# ============================================================================
# Production stage
# ============================================================================
FROM base as production

WORKDIR /app

# Set production environment
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Copy production dependencies
COPY --from=prod-deps /app/node_modules ./node_modules

# Copy built application from builder stage
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json

# Copy static assets and configuration
COPY nginx.conf* ./

# Create necessary directories
RUN mkdir -p /app/logs

# Change ownership to nodejs user
RUN chown -R nodejs:nodejs /app

# Switch to non-root user
USER nodejs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/ || exit 1

# Expose port
EXPOSE 3000

# Production command
CMD ["npm", "start"]

# ============================================================================
# Nginx production stage (optional)
# ============================================================================
FROM nginx:alpine as nginx-production

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy custom nginx site configuration
COPY nginx-site.conf /etc/nginx/conf.d/default.conf

# Health check for nginx
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

# ============================================================================
# Testing stage
# ============================================================================
FROM development as testing

# Install additional testing dependencies if needed
RUN npm install --save-dev \
    @testing-library/jest-dom \
    @testing-library/react \
    @testing-library/user-event

# Copy test configuration
COPY jest.config.js ./
COPY setupTests.js ./

# Run tests by default
CMD ["npm", "test", "--", "--coverage", "--watchAll=false"]

# ============================================================================
# Build info and labels
# ============================================================================

# Add metadata labels to production stage
LABEL maintainer="{{cookiecutter.author_name}} <{{cookiecutter.author_email}}>" \
      version="{{cookiecutter.version}}" \
      description="{{cookiecutter.description}} - Frontend" \
      org.opencontainers.image.title="{{cookiecutter.project_name}} Frontend" \
      org.opencontainers.image.description="{{cookiecutter.description}} - React Frontend" \
      org.opencontainers.image.version="{{cookiecutter.version}}" \
      org.opencontainers.image.vendor="{{cookiecutter.author_name}}" \
      org.opencontainers.image.source="https://github.com/{{cookiecutter.github_username}}/{{cookiecutter.project_slug}}" \
      org.opencontainers.image.licenses="{{cookiecutter.license}}"

# ============================================================================
# Multi-stage build targets summary:
# - base: Common base layer with Node.js and system dependencies
# - deps: All dependencies (dev + prod)
# - development: Development environment with hot reload
# - builder: Builds the production application
# - prod-deps: Production dependencies only
# - production: Final production image with built app
# - nginx-production: Nginx-served static files (alternative)
# - testing: Testing environment with test dependencies
# ============================================================================