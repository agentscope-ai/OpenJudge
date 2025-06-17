from setuptools import find_packages, setup

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ""

setup(
    name="rm-gallery",
    version="0.1.0",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="",
    author_email="",
    url="",
    packages=find_packages(),
    python_requires="==3.10",
    install_requires=[
        "pandas>=2.2.3,<3.0.0",
        "pytest>=8.3.5,<9.0.0",
        "pyyaml>=6.0.2,<7.0.0",
        "loguru>=0.7.3,<0.8.0",
        "jsonlines>=4.0.0,<5.0.0",
        "transformers>=4.52.4,<5.0.0",
        "pydantic>=2.11.5,<3.0.0",
        "openai>=1.85.0,<2.0.0",
        "tiktoken>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.5,<9.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    include_package_data=True,
    zip_safe=False,
)
