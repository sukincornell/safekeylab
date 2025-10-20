"""
Aegis - Divine Protection for AI Systems
Enterprise-grade privacy shield inspired by Zeus's legendary protection
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="aegis",
    version="1.0.0",
    author="Aegis Security",
    author_email="support@aegis-shield.ai",
    description="Aegis: Divine shield for AI privacy - Military-grade PII protection with <50ms latency",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aegis-shield/aegis",
    project_urls={
        "Documentation": "https://docs.aegis-shield.ai",
        "Bug Tracker": "https://github.com/aegis-shield/aegis/issues",
        "Source Code": "https://github.com/aegis-shield/aegis",
        "Changelog": "https://github.com/aegis-shield/aegis/blob/main/CHANGELOG.md",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Security",
        "Topic :: Security :: Cryptography",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="aegis privacy ai pii data-protection shield differential-privacy k-anonymity gdpr ccpa hipaa security anonymization enterprise",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
        ],
        "all": [
            "torch[cuda]>=2.0.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aegis=aegis.__init__:main",
            "aegis-benchmark=aegis.privacy_benchmark:main",
        ],
    },
    include_package_data=True,
    package_data={
        "aegis": ["*.json", "*.yaml", "*.txt", "*.md"],
    },
    zip_safe=False,
)