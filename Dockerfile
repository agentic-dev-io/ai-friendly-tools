# AIFT-OS: AI-Friendly Tools Operating System v2026
# A modern, AI-ready container with Claude Code and all AIFT tools
# Updated with latest dependencies (Jan 2026)
# Includes: Pydantic 2.12.5, Typer 0.21.1, Rich 14.2.0, DuckDB 1.4.3, Ruff 0.14.13

# ============================================================================
# Builder Stage: Compile dependencies and tools
# ============================================================================
FROM python:3.11-slim-bookworm AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    ca-certificates \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install UV (Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Install Bun (JavaScript runtime)
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:$PATH"

# Install Rust for CLI tools
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Install modern CLI tools (latest versions)
RUN cargo install ripgrep fd-find bat eza

# ============================================================================
# Runtime Stage: Minimal production image
# ============================================================================
FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="AIFT-OS" \
      org.opencontainers.image.description="AI-Friendly Tools Operating System with Claude Code" \
      org.opencontainers.image.version="2026-01" \
      org.opencontainers.image.authors="Björn Bethge <bjoern.bethge@gmail.com>" \
      org.opencontainers.image.source="https://github.com/bjoernbethge/ai-friendly-tools" \
      org.opencontainers.image.created="2026-01-17"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core utilities
    bubblewrap \
    git \
    curl \
    wget \
    ca-certificates \
    unzip \
    # JSON/YAML processing
    jq \
    # Required for some Python packages
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

# Install yq (YAML processor)
RUN wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq

# Create non-root user
RUN useradd -m -s /bin/bash -u 1000 aift && \
    mkdir -p /workspace /home/aift/.config /home/aift/aift/logs /home/aift/.local/bin && \
    chown -R aift:aift /workspace /home/aift

# Copy Rust CLI tools from builder (before USER switch, but to user-writable location)
COPY --from=builder --chown=aift:aift /root/.cargo/bin/rg /home/aift/.local/bin/
COPY --from=builder --chown=aift:aift /root/.cargo/bin/fd /home/aift/.local/bin/
COPY --from=builder --chown=aift:aift /root/.cargo/bin/bat /home/aift/.local/bin/
COPY --from=builder --chown=aift:aift /root/.cargo/bin/eza /home/aift/.local/bin/

# Switch to non-root user
USER aift
WORKDIR /home/aift

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/aift/.local/bin:$PATH"

# Install Bun
RUN curl -fsSL https://bun.sh/install | bash
ENV BUN_INSTALL="/home/aift/.bun"
ENV PATH="$BUN_INSTALL/bin:$PATH"

# Create node symlink for bun (for npm packages that expect node)
RUN ln -s /home/aift/.bun/bin/bun /home/aift/.bun/bin/node

# Install DuckDB CLI
RUN wget https://github.com/duckdb/duckdb/releases/latest/download/duckdb_cli-linux-amd64.zip && \
    unzip duckdb_cli-linux-amd64.zip && \
    mv duckdb /home/aift/.local/bin/ && \
    chmod +x /home/aift/.local/bin/duckdb && \
    rm duckdb_cli-linux-amd64.zip

# Install Claude Code and Sandbox Runtime
RUN bun add -g @anthropic-ai/claude-code @anthropic-ai/sandbox-runtime

# Install Serena Coding Agent Toolkit
RUN uv tool install git+https://github.com/oraios/serena.git

# Install OpenCode AI Coding Agent
# OpenCode is an open-source AI coding agent (https://opencode.ai)
# Supports Claude Code integration, TUI, and programmatic API access
RUN wget https://github.com/anomalyco/opencode/releases/download/v1.1.25/opencode-linux-x64.tar.gz && \
    tar -xzf opencode-linux-x64.tar.gz && \
    mv opencode /home/aift/.local/bin/ && \
    chmod +x /home/aift/.local/bin/opencode && \
    rm opencode-linux-x64.tar.gz

# Copy AIFT workspace
COPY --chown=aift:aift . /workspace/aift
WORKDIR /workspace/aift

# Install AIFT tools
RUN uv sync

# Set up environment
ENV PYTHONUNBUFFERED=1 \
    AIFT_LOG_LEVEL=INFO \
    PATH="/home/aift/.local/bin:$PATH" \
    # Claude Code configuration
    CLAUDE_CODE_SANDBOX=true

# Create convenience aliases
RUN echo 'alias ls="eza --icons"' >> /home/aift/.bashrc && \
    echo 'alias ll="eza -la --icons"' >> /home/aift/.bashrc && \
    echo 'alias cat="bat"' >> /home/aift/.bashrc && \
    echo 'alias find="fd"' >> /home/aift/.bashrc && \
    echo 'alias grep="rg"' >> /home/aift/.bashrc && \
    echo '' >> /home/aift/.bashrc && \
    echo '# AIFT-OS Welcome' >> /home/aift/.bashrc && \
    echo 'echo "🚀 AIFT-OS - AI-Friendly Tools OS"' >> /home/aift/.bashrc && \
    echo 'uv tool list' >> /home/aift/.bashrc && \
    echo 'echo ""' >> /home/aift/.bashrc

# Set working directory
WORKDIR /workspace

# Expose common ports (optional)
EXPOSE 8000 8080 9999

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD uv run python -c "from core import __version__; print(__version__)" || exit 1

# Default entrypoint
ENTRYPOINT ["/bin/bash", "--login"]
