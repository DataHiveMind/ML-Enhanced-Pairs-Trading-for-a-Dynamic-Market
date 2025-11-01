# Contributing to ML-Enhanced Pairs Trading

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ML-Enhanced-Pairs-Trading-for-a-Dynamic-Market.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Set up your development environment (see below)

## Development Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]  # Install development dependencies
   ```

3. Copy environment template:
   ```bash
   cp .env.example .env
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and modular
- Add comments for complex logic

### Running Code Formatters

```bash
# Format code with black
black src/ tests/

# Check code style with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

## Testing

- Write unit tests for all new features
- Ensure all tests pass before submitting a PR
- Aim for high test coverage (>80%)

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_utils.py
```

## Making Changes

1. Make your changes in your feature branch
2. Add or update tests as needed
3. Update documentation if necessary
4. Ensure all tests pass
5. Commit your changes with clear, descriptive messages

### Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters or less
- Reference issues and pull requests when relevant

Example:
```
Add cointegration test for multiple time windows

- Implement rolling window cointegration analysis
- Add unit tests for new functionality
- Update documentation

Fixes #123
```

## Submitting a Pull Request

1. Push your changes to your fork
2. Create a pull request from your fork to the main repository
3. Provide a clear description of the changes
4. Link any related issues
5. Wait for review and address any feedback

### Pull Request Checklist

- [ ] Code follows the project's style guidelines
- [ ] All tests pass
- [ ] New code has appropriate test coverage
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] Changes are focused and atomic

## Project Structure

```
src/                    # Core source code
tests/                  # Unit tests
notebooks/              # Jupyter notebooks for exploration
docs/                   # Documentation
configs/                # Configuration files
```

## Areas for Contribution

We welcome contributions in the following areas:

- **Algorithm improvements**: Better cointegration tests, ML models, trading strategies
- **Performance optimization**: Faster backtesting, efficient data processing
- **Documentation**: Tutorials, examples, API documentation
- **Testing**: Increase test coverage, add integration tests
- **Bug fixes**: Fix reported issues
- **New features**: Propose and implement new functionality

## Questions or Issues?

- Open an issue for bug reports or feature requests
- Start a discussion for questions or ideas
- Review existing issues before creating a new one

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🚀
