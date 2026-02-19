"""
ModularForge — Setup Script
============================
Installs ModularForge as a local editable package so that all internal
imports (e.g. `from modularforge.model.expert import ExpertFFN`) work
seamlessly from any script or notebook.

Usage:
    cd /path/to/LLM_In_Parts
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="modularforge",
    version="0.1.0",
    author="Aditya",
    description=(
        "ModularForge: Memory-Bounded Modular Training and Zero-Retraining "
        "Assembly of Large Language Models on Resource-Constrained Hardware"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aditya/ModularForge",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "safetensors>=0.4.0",
        "datasets>=2.14.0",
        "tokenizers>=0.15.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "tensorboard>=2.14.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
