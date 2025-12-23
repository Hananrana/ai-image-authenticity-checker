# Contributing to AI Image Authenticity Checker

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-image-authenticity-checker.git
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate   # Windows
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Development Setup

### GPU Training
For GPU training, install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Running Tests
```bash
pytest tests/ -v
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Add docstrings to all public functions and classes
- Keep lines under 100 characters

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests to ensure nothing is broken
4. Commit with clear messages: `git commit -m "feat: Add new feature"`
5. Push to your fork: `git push origin feature/your-feature`
6. Open a Pull Request

## Commit Message Format

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

## Questions?

Open an issue for questions or suggestions!
