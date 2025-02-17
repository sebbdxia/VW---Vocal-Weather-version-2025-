from setuptools import setup, find_packages

setup(
    name="VW_Vocal_Weather",
    version="2025.0",
    packages=find_packages(),
    install_requires=[
        "azure-cognitiveservices-speech",
        "httpx",
        "spacy",
        "uvicorn",
        "streamlit",
        "fastapi",
        "python-dotenv",
        # ...autres dépendances nécessaires...
    ],
)
