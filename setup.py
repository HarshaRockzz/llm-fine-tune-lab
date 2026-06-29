from setuptools import setup, find_packages

setup(
    name="llm_fine_tune_lab",
    version="1.0.0",
    description="Production-grade LLM fine-tuning lab with LoRA/QLoRA, vLLM serving, and evaluation harness",
    author="Harshavardhan",
    python_requires=">=3.11",
    packages=find_packages(where=".", include=["src", "src.*"]),
    package_dir={"": "."},
    install_requires=[
        "pydantic>=2.7.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "training": open("requirements-training.txt").read().splitlines(),
        "inference": open("requirements-inference.txt").read().splitlines(),
        "streamlit": open("streamlit_requirements.txt").read().splitlines(),
        "dev": open("requirements.txt").read().splitlines(),
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
