# ADR-004: PostgreSQL Database Selection

## Status
Accepted

## Date
2025-07-27

## Context
We need a database solution for the Agentic Startup Studio that provides:
- ACID compliance for data integrity
- Advanced querying capabilities
- JSON/NoSQL hybrid support
- Full-text search capabilities
- Scalability for growing data needs
- Strong ecosystem support
- Vector database capabilities for AI embeddings

## Decision
We have chosen PostgreSQL 15+ as our primary database.

## Rationale
1. **ACID Compliance**: Strong consistency and reliability for critical data
2. **Advanced Features**: JSON support, full-text search, array types
3. **Vector Extensions**: pgvector extension for AI embeddings and similarity search
4. **Performance**: Excellent query optimization and indexing capabilities
5. **Ecosystem**: Mature ecosystem with extensive tooling and extensions
6. **Scalability**: Read replicas, partitioning, and horizontal scaling options
7. **Open Source**: No vendor lock-in with strong community support

## Alternatives Considered
- **MongoDB**: NoSQL flexibility but lacks ACID guarantees
- **MySQL**: Simpler but less advanced features for AI workloads
- **Redis**: In-memory speed but not suitable for persistent primary data
- **Vector Databases**: Specialized but would require additional primary database

## Consequences

### Positive
- Strong data consistency and reliability
- Advanced querying capabilities for complex data relationships
- Native JSON support for flexible schema evolution
- Vector search capabilities for AI embeddings
- Excellent performance with proper indexing

### Negative
- More complex than simpler databases
- Requires database administration knowledge
- Higher resource usage than simpler alternatives

## Implementation Details
- PostgreSQL 15+ with pgvector extension
- SQLAlchemy ORM with async support
- Alembic for database migrations
- Connection pooling for performance
- Read replicas for scaling read operations

## Schema Design Principles
```sql
-- Example table with AI-specific features
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(1536), -- For OpenAI embeddings
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Vector similarity index
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);

-- Full-text search index
CREATE INDEX documents_content_fts ON documents USING gin(to_tsvector('english', content));
```

## Performance Optimization
- Proper indexing strategy for common queries
- Query optimization and EXPLAIN analysis
- Connection pooling with pgbouncer
- Partitioning for large tables
- Regular VACUUM and ANALYZE operations

## Backup and Recovery
- Automated daily backups with point-in-time recovery
- Cross-region backup replication
- Regular recovery testing
- Database monitoring and alerting

## Security
- SSL/TLS encryption for connections
- Row-level security for multi-tenant applications
- Regular security updates
- Database audit logging

## Monitoring
- Query performance monitoring
- Connection and resource usage tracking
- Slow query logging and analysis
- Database health metrics

## Compliance
This decision supports our requirements for:
- Reliable data storage with ACID guarantees
- Advanced querying for complex AI workflows
- Vector search for AI embeddings
- Scalability for growing applications