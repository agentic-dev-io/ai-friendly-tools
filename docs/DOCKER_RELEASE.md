# Docker Release Process

Guide for building and publishing Docker images for AIFT.

## Prerequisites

- Docker installed and running
- Docker Hub account or GitHub Container Registry access
- GitHub repository with Actions enabled
- Secrets configured in GitHub

## GitHub Secrets Configuration

Set up these secrets in your GitHub repository settings:

### For Docker Hub

1. Go to: **Settings > Secrets and variables > Actions**
2. Add secrets:
   - `DOCKER_USERNAME`: Your Docker Hub username
   - `DOCKER_PASSWORD`: Your Docker Hub token (not password!)

### For GitHub Container Registry

Uses `GITHUB_TOKEN` automatically (no additional setup needed)

## Building Locally

### Build for Current Platform

```bash
docker build -t aift-os:latest .
```

### Build for Multiple Platforms

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t myusername/aift-os:latest \
  --push .
```

### Build with Custom Tag

```bash
docker build -t aift-os:2026-01 .
```

## Publishing to Docker Hub

### Automatic (GitHub Actions)

1. Tag a release in Git:
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

2. The `docker-release.yml` workflow will:
   - Run tests and validation
   - Build the image
   - Push to Docker Hub: `your-username/aift-os:0.2.0`
   - Push to GitHub Container Registry
   - Create a GitHub release

### Manual

```bash
# Login to Docker Hub
docker login

# Tag image
docker tag aift-os:latest myusername/aift-os:0.2.0
docker tag aift-os:latest myusername/aift-os:latest

# Push to Docker Hub
docker push myusername/aift-os:0.2.0
docker push myusername/aift-os:latest
```

## Publishing to GitHub Container Registry

### Automatic (GitHub Actions)

Same as Docker Hub - triggered by version tags.

### Manual

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USER --password-stdin

# Tag image
docker tag aift-os:latest ghcr.io/your-org/aift-os:0.2.0
docker tag aift-os:latest ghcr.io/your-org/aift-os:latest

# Push to GHCR
docker push ghcr.io/your-org/aift-os:0.2.0
docker push ghcr.io/your-org/aift-os:latest
```

## Semantic Versioning

AIFT follows semantic versioning:

- **MAJOR**: Breaking changes (v1.0.0)
- **MINOR**: New features (v0.2.0)
- **PATCH**: Bug fixes (v0.1.1)

### Creating a Release

```bash
# Create and push a tag
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0

# GitHub Actions will automatically:
# 1. Run all tests
# 2. Build Docker image
# 3. Push to registries
# 4. Create GitHub release with images tagged as:
#    - v0.2.0
#    - 0.2 (latest minor)
#    - 0 (latest major)
#    - latest
#    - commit SHA
```

## Image Tags

Published images include multiple tags:

```
docker pull myusername/aift-os:v0.2.0      # Exact version
docker pull myusername/aift-os:0.2          # Latest patch
docker pull myusername/aift-os:0            # Latest minor
docker pull myusername/aift-os:latest       # Latest release
docker pull myusername/aift-os:main-abc123  # Branch + commit
```

## CI/CD Pipeline

The Docker release workflow includes:

1. **Code Quality Checks**
   - Ruff linting
   - MyPy type checking
   - Test execution
   - Coverage reporting

2. **Security Scanning**
   - Trivy vulnerability scanning
   - CodeQL analysis
   - Dependency checks

3. **Build & Push**
   - Build multi-platform images (amd64, arm64)
   - Push to Docker Hub
   - Push to GitHub Container Registry
   - Build cache optimization

4. **Release Creation**
   - Automatic GitHub release
   - Release notes with image references
   - Changelog links

## Dockerfile Details

The Dockerfile uses:

- **Base**: `python:3.11-slim-bookworm`
- **Size**: ~1.04 GB
- **Includes**:
  - All AIFT tools and dependencies
  - CLI tools: rg, fd, bat, eza
  - ML/AI libraries: torch, transformers, scipy, numpy
  - Health check every 30 seconds

### Health Check

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD aift validate || exit 1
```

## Running Published Images

### From Docker Hub

```bash
docker pull your-username/aift-os:latest
docker run -it your-username/aift-os:latest aift --help
```

### From GitHub Container Registry

```bash
docker pull ghcr.io/your-org/aift-os:latest
docker run -it ghcr.io/your-org/aift-os:latest aift --help
```

### With Volume Mounts

```bash
docker run -it \
  -v ~/.aift:/root/.aift \
  -v ./data:/app/data \
  your-username/aift-os:latest \
  aift info
```

### With Environment Variables

```bash
docker run -it \
  -e AIFT_LOG_LEVEL=DEBUG \
  your-username/aift-os:latest \
  aift debug
```

## Docker Compose

Use `docker-compose.yml` for complete stack:

```bash
docker-compose up -d
docker-compose logs -f aift
docker-compose down
```

## Troubleshooting

### Build Fails

1. Check Docker is running: `docker ps`
2. Clean build cache: `docker builder prune`
3. Check disk space: `df -h`

### Push Fails

1. Verify credentials: `docker login`
2. Check repository access: `docker push test-image`
3. Verify secrets in GitHub Actions

### Image Too Large

- Clear build cache: `docker system prune -a`
- Use `.dockerignore` to exclude files
- Use multi-stage builds (see Dockerfile)

### Health Check Fails

- Ensure `aift` command is in PATH
- Check log level doesn't prevent success output
- Verify Python installation in container

## Best Practices

1. **Always tag releases** with semantic versions
2. **Run tests locally** before pushing
3. **Use specific base image versions** (not `latest`)
4. **Enable security scanning** in CI/CD
5. **Document breaking changes** in release notes
6. **Test multi-platform builds** before release
7. **Monitor image size** to optimize distribution

## Security

- Images are scanned with Trivy for vulnerabilities
- CodeQL analyzes code for security issues
- Only push from authenticated CI/CD
- Use token-based authentication (not passwords)
- Regularly update base image

## Support

For issues with Docker releases:
- Check GitHub Actions logs
- Review Dockerfile
- Verify secrets are set
- Test locally first

See `/docs/DOCKER.md` for more Docker usage information.
