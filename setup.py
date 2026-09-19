from setuptools import find_packages, setup

setup(
    name="open-sair",
    version="1.0.0",
    description="Open Santali AI Research: Unified Multimodal Sovereign AI Framework for Ol Chiki",
    author="Open SAIR Research Consortium",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
