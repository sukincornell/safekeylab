# Contributing to Grey

Thank you for your interest in contributing to Grey! We welcome contributions from the community to help make privacy preservation better for everyone.

## How to Contribute

### Reporting Issues

1. Check if the issue already exists in our [issue tracker](https://github.com/grey-ai/grey/issues)
2. If not, create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, Grey version)

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Format code with Black (`black grey/`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to your branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/grey.git
cd grey

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
flake8 grey/
mypy grey/

# Format code
black grey/
```

## Code Standards

- Follow PEP 8 style guide
- Use type hints for all functions
- Write docstrings for all public functions
- Maintain test coverage above 80%
- Keep functions focused and under 50 lines
- Use meaningful variable names

## Testing

- Write unit tests for new features
- Include integration tests for complex workflows
- Test edge cases and error conditions
- Benchmark performance for critical paths

## Documentation

- Update docstrings for API changes
- Add examples for new features
- Update README if needed
- Add entries to CHANGELOG.md

## Privacy and Security

Given the nature of this project, please:
- Never commit real PII or sensitive data
- Use synthetic data for tests
- Consider privacy implications of all changes
- Follow security best practices

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing opinions

## Recognition

Contributors will be recognized in:
- The project's AUTHORS file
- Release notes for significant contributions
- Annual contributor spotlight (for regular contributors)

## Questions?

Feel free to:
- Open a discussion in [GitHub Discussions](https://github.com/grey-ai/grey/discussions)
- Reach out on our [Discord server](https://discord.gg/grey-ai)
- Email us at contribute@grey-ai.com

Thank you for helping make Grey better! 🎉