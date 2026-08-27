python -m venv venv


venv\Scripts\activate


pip install -r requirements.txt



python extrator_conteudo.py


python criar_rag.py



Em terminais separados:

uvicorn app_mms:app --reload --port 8000



uvicorn app_llm:app --reload --port 8001
