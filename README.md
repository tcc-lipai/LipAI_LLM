python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python extrator_conteudo.py
python criar_rag.py
Em terminais separados:
uvicorn app_mms.py:app --reload
uvicorn app_llm.py:app --reload
