from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Elasticsearch settings
    ELASTICSEARCH_HOST: str = "localhost"
    ELASTICSEARCH_PORT: int = 9200
    ELASTICSEARCH_INDEX_PREFIX: str = "sanskrit"
    
    # Model paths and settings
    BERT_MODEL_NAME: str = "bert-base-multilingual-cased"
    SENTENCE_TRANSFORMER_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # API settings
    SANSKRIT_HERITAGE_API_URL: str = "https://heritage.samsaadhanii.in"
    UOH_TAGGER_URL: Optional[str] = None
    
    # Processing settings
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 32
    NUM_RETRIEVAL_CANDIDATES: int = 100
    
    # Training settings
    LEARNING_RATE: float = 2e-5
    NUM_EPOCHS: int = 3
    WARMUP_STEPS: int = 500
    
    class Config:
        env_file = ".env"

settings = Settings()