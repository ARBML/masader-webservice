from os import environ


class Config:
    REDIS_URL = environ.get('REDIS_URL', 'redis://localhost:6379')

    HF_SECRET_KEY = environ.get('HF_SECRET_KEY')

    GH_SECRET_KEY = environ.get('GH_SECRET_KEY')

    REFRESH_PASSWORD = environ.get('password')

    API_KEY = environ.get('API_KEY')
    API_URL = environ.get('API_URL', 'https://openrouter.ai/api/v1/chat/completions')
    MODEL_NAME = environ.get('MODEL_NAME', 'google/gemini-2.5-flash-lite')

    # Which Retriever the chat endpoint uses. See utils.retrieval.get_retriever
    # for the available algorithms (bm25, sql).
    RETRIEVAL_ALGORITHM = environ.get('RETRIEVAL_ALGORITHM', 'sql')

    # Schema endpoint used by the SQL retriever (POST with name=ar).
    MASADER_SCHEMA_URL = environ.get('MASADER_SCHEMA_URL', 'https://mextract-production.up.railway.app/schema')

    # Agent that decides whether a turn needs a fresh retrieval. 'llm' classifies
    # follow-ups to reuse prior datasets; 'always' retrieves every turn. See
    # utils.router.ROUTERS.
    RETRIEVAL_ROUTER = environ.get('RETRIEVAL_ROUTER', 'llm')
