from setuptools import setup, find_packages

setup(
    name="hello_pulse_rag",  
    version="0.1.0",  
    author="O. A. Stéphane KPOVIESSI",
    author_email="oastephaneamiche@gmail.com",
    description="Retrieval-Augmented Generation utilities for Hello Pulse",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/hello_pulse_rag",  
    license="MIT",  
    packages=find_packages(), 
    include_package_data=True,
    install_requires=[
        "torch>=2.5.0",
        "sentence-transformers>=2.2.2",
        "TTS>=0.22.0",
        "fastapi>=0.115.6",
        "uvicorn>=0.34.0",
        "pdfplumber",
        "python-docx",
        "beautifulsoup4",
        "chardet>=5.1.0",
        "python-multipart",
        "spacy>=3.8.3",
        "pydantic>=2.10.4",
        "langdetect>=1.0.9",
        "langid>=1.1.6",
        "json-log-formatter",
        "huggingface_hub>=0.16.4",
        "playsound>=1.3.0",
    ],
    extras_require={
        "gpu": [
            "--extra-index-url https://download.pytorch.org/whl/cu124",
            "torch>=2.5.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "rag-utils=ragutils.main:main", 
        ],
    },
)
