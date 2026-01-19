# AIFT Custom Copilot Agents

This directory contains specialized GitHub Copilot agents tailored for the AIFT (AI-Friendly Tools) project. These agents provide domain-specific expertise to help maintain code quality, consistency, and best practices across the project.

## Available Agents

### 🐍 [Python Expert](agents/python-expert.agent.md)
**Specialization**: Python 3.11+ development, UV workspace management, type hints, and code quality

**Use When:**
- Writing or refactoring Python code
- Managing dependencies across packages
- Implementing type hints and Pydantic models
- Working with UV workspace structure
- Ensuring code quality and PEP 8 compliance

**Key Expertise:**
- Python 3.11+ features (pattern matching, exception groups)
- UV workspace with core/web/mcp-manager/memo packages
- Type hints and MyPy strict mode
- Pydantic models and validation
- Ruff linting and formatting

---

### 🖥️ [CLI Specialist](agents/cli-specialist.agent.md)
**Specialization**: Building command-line interfaces with Typer and Rich

**Use When:**
- Creating new CLI commands
- Improving terminal output and formatting
- Implementing argument parsing and validation
- Designing user-friendly command interfaces
- Adding progress bars, tables, or rich formatting

**Key Expertise:**
- Typer framework for CLI development
- Rich terminal formatting and UI
- CLI best practices and conventions
- Error handling and exit codes
- Help text and documentation

---

### 🧪 [Testing Specialist](agents/testing-specialist.agent.md)
**Specialization**: Pytest, test coverage, and testing patterns

**Use When:**
- Writing unit tests for new features
- Improving test coverage
- Implementing fixtures and mocking
- Testing CLI commands
- Setting up parametrized tests

**Key Expertise:**
- Pytest framework and fixtures
- Mocking and patching
- Test organization and structure
- Coverage analysis
- CLI testing with Typer

---

### 🐳 [Docker Specialist](agents/docker-specialist.agent.md)
**Specialization**: Docker, docker-compose, and container optimization

**Use When:**
- Modifying Dockerfile or docker-compose.yml
- Optimizing container images
- Managing multi-service deployments
- Setting up development environments
- Configuring production containers

**Key Expertise:**
- Multi-stage Docker builds
- Image size optimization
- docker-compose orchestration
- Security best practices
- Volume management and persistence

---

### 📚 [Documentation Writer](agents/documentation-writer.agent.md)
**Specialization**: Technical documentation, READMEs, and user guides

**Use When:**
- Creating or updating README files
- Writing API documentation
- Creating tutorials and guides
- Documenting configuration options
- Maintaining changelog

**Key Expertise:**
- Markdown documentation
- Code documentation (docstrings)
- API reference documentation
- Tutorial and guide structure
- AI-friendly documentation formats

---

### 🦆 [DuckDB Expert](agents/duckdb-expert.agent.md)
**Specialization**: DuckDB database design, queries, and optimization

**Use When:**
- Designing database schemas
- Writing SQL queries
- Implementing semantic search
- Optimizing query performance
- Working with MCP Manager database

**Key Expertise:**
- DuckDB schema design
- Python-DuckDB integration
- Vector embeddings and semantic search
- Query optimization and indexing
- JSON operations in DuckDB

---

## How to Use Agents

### In GitHub Copilot Coding Agent (CCA)
1. When creating or assigning an issue to Copilot
2. Select the appropriate specialized agent from the list
3. The agent will apply its domain expertise to the task

### In VS Code
1. Open Copilot Chat (@workspace)
2. Reference agents in your conversations
3. Ask domain-specific questions to leverage agent expertise

### Best Practices

#### Choose the Right Agent
- **Python Code Changes**: Use Python Expert
- **CLI Development**: Use CLI Specialist
- **Writing Tests**: Use Testing Specialist
- **Docker Changes**: Use Docker Specialist
- **Documentation Updates**: Use Documentation Writer
- **Database Work**: Use DuckDB Expert

#### Combine Agents When Needed
For complex tasks that span multiple domains:
1. Start with the primary agent for the main work
2. Consult other agents for related aspects
3. Example: Use Python Expert for code, then Testing Specialist for tests

#### Agent Workflow
```
Problem → Select Agent → Agent Applies Expertise → Review → Iterate
```

## Agent Development Guidelines

### Creating New Agents

When adding a new specialized agent to this project:

1. **Identify the Domain**: Determine a specific area that needs specialized knowledge
2. **Create Agent File**: Name it descriptively (e.g., `security-expert.agent.md`)
3. **Define Scope**: Clearly specify what the agent does and doesn't handle
4. **Document Expertise**: List the agent's key areas of knowledge
5. **Provide Examples**: Include code samples and usage patterns
6. **Update This File**: Add the new agent to the list above

### Agent File Structure

```markdown
---
name: Agent Name
description: Brief description of agent specialization
---

# Agent Name

Introduction and overview of the agent's role.

## Core Expertise
- Key skill 1
- Key skill 2
- Key skill 3

## Domain Knowledge
Detailed information about the domain.

## Best Practices
Guidelines and patterns to follow.

## Quality Checklist
- [ ] Checklist item 1
- [ ] Checklist item 2

## Success Criteria
Clear definition of successful completion.
```

## Project Context

### AIFT Architecture
```
ai-friendly-tools/
├── core/              # Shared library (config, logging, CLI)
├── web/               # Web intelligence suite
├── mcp-manager/       # DuckDB MCP manager with semantic search
├── memo/              # Memory & AI tools (not in workspace)
├── tests/             # All tests at repository root
└── docs/              # Documentation
```

### Key Technologies
- **Python**: 3.11+ with type hints
- **UV**: Fast Python package manager
- **Typer**: CLI framework
- **Rich**: Terminal formatting
- **DuckDB**: Embedded database
- **Pydantic**: Data validation
- **Pytest**: Testing framework
- **Ruff**: Linting and formatting
- **Docker**: Containerization

### Development Commands
```bash
# Setup
uv sync --all-groups

# Testing
uv run pytest tests/ -v

# Linting
uv run ruff check .

# Formatting
uv run ruff format .

# Type Checking
uv run mypy core/src web/src mcp-manager/src --strict

# Docker
docker-compose up -d
docker-compose exec aift-os bash
```

## Contributing to Agents

### Improving Existing Agents
1. Review the agent's current content
2. Identify gaps or outdated information
3. Update with latest project practices
4. Test the agent with real tasks
5. Submit a PR with improvements

### Suggesting New Agents
1. Check if the domain overlaps with existing agents
2. Create an issue describing the proposed agent
3. Explain why a specialized agent is needed
4. List the key expertise areas
5. Get feedback from maintainers

## Feedback and Issues

If you encounter issues with agents or have suggestions:
1. Check existing issues in the repository
2. Create a new issue with:
   - Agent name
   - Issue description
   - Expected vs actual behavior
   - Suggestions for improvement

## References

- [GitHub Copilot Custom Agents Documentation](https://docs.github.com/en/copilot/how-tos/use-copilot-agents/coding-agent/create-custom-agents)
- [AIFT Project Documentation](../docs/)
- [AIFT Architecture Guide](../docs/ARCHITECTURE.md)
- [AIFT Development Guide](../docs/DEVELOPMENT.md)

---

**Last Updated**: 2026-01-19
**Version**: 1.0.0
