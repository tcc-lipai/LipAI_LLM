import json
import os

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from adaptive_engine import engine_adaptativa

try:
    import pyphen
    _HIFENIZADOR = pyphen.Pyphen(lang="pt_BR")
except ImportError:
    _HIFENIZADOR = None

app = FastAPI(title="LipAI LLM API - Lições de Fonemas", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PASTA_CHROMA = "./chroma_db"
CAMINHO_BANCO_CONTEUDO = "./banco_conteudo.json"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=PASTA_CHROMA, embedding_function=embeddings)
modelo_llm = OllamaLLM(model="qwen2.5:3b", temperature=0.0, num_predict=60)

MMS_API_URL = os.getenv("MMS_API_URL", "http://127.0.0.1:8000/avaliar-fala-alinhada/")


def _usuario_padrao(nome: str = "Aluno"):
    return {
        "nome": nome,
        "fonema_atual": "A",
        "nivel_atual": 1,
        "passo_licao": 0,
        "scores_licao": [],
        "historico_itens": {},
        "historico_tentativas": [],
    }


USUARIOS_DB = {1: _usuario_padrao("Criança Teste")}


def carregar_banco_conteudo() -> dict:
    if not os.path.exists(CAMINHO_BANCO_CONTEUDO):
        print(
            f"⚠️  {CAMINHO_BANCO_CONTEUDO} não encontrado. "
            "Rode `python extrator_conteudo.py` antes de subir a API."
        )
        return {}
    with open(CAMINHO_BANCO_CONTEUDO, "r", encoding="utf-8") as f:
        return json.load(f)


BANCO_CONTEUDO = carregar_banco_conteudo()


def formatar_para_nivel(item: str, nivel: int) -> str:
    if nivel == 1:
        return f"{item} {item}"
    if nivel == 2 and _HIFENIZADOR:
        return _HIFENIZADOR.inserted(item).upper()
    return item.upper()


async def chamar_mms_gop(texto_esperado: str, arquivo: UploadFile):
    try:
        conteudo_audio = await arquivo.read()
        async with httpx.AsyncClient() as cliente:
            files = {"arquivo": (arquivo.filename, conteudo_audio, "audio/wav")}
            data = {"texto_esperado": texto_esperado}
            res = await cliente.post(MMS_API_URL, data=data, files=files, timeout=60.0)
            if res.status_code != 200:
                raise HTTPException(status_code=500, detail="Erro retornado pela MMS.")
            return res.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro de conexão com MMS: {str(e)}")


@app.post("/treino/selecionar-nivel/{usuario_id}")
def selecionar_nivel_manual(usuario_id: int, nivel: int = Query(..., ge=1, le=4), fonema: str = Query("A")):
    if usuario_id not in USUARIOS_DB:
        USUARIOS_DB[usuario_id] = _usuario_padrao(f"Aluno {usuario_id}")

    usuario = USUARIOS_DB[usuario_id]
    usuario["nivel_atual"] = nivel
    usuario["fonema_atual"] = fonema.upper()
    usuario["passo_licao"] = 0
    usuario["scores_licao"] = []
    return {"status": "sucesso", "usuario": usuario}


@app.get("/treino/dica-fonema/{usuario_id}")
async def obter_dica_fonema(usuario_id: int):
    usuario = USUARIOS_DB.get(usuario_id, {"fonema_atual": "A"})
    fonema = usuario["fonema_atual"]

    termo_busca = f"posição dos lábios e da língua, passagem do ar, treino visual do fonema {fonema}"
    resultados = await run_in_threadpool(
        db.similarity_search, termo_busca, k=2, filter={"fonema": fonema}
    )
    contexto_dica = "\n\n".join([d.page_content for d in resultados]) if resultados else ""

    if not contexto_dica:
        contexto_dica = f"Posicione os lábios e a língua de forma relaxada para emitir o som do fonema {fonema} com clareza."

    return {"fonema": fonema, "dica": contexto_dica}


@app.get("/treino/proximo-exercicio/{usuario_id}")
async def obter_proximo_exercicio(usuario_id: int, prova: bool = False):
    if usuario_id not in USUARIOS_DB:
        USUARIOS_DB[usuario_id] = _usuario_padrao(f"Aluno {usuario_id}")

    usuario = USUARIOS_DB[usuario_id]
    fonema = usuario["fonema_atual"]
    nivel = usuario["nivel_atual"]
    passo = usuario.get("passo_licao", 0)

    item = engine_adaptativa.proximo_item(usuario, BANCO_CONTEUDO, modo_prova=prova)

    if item is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Nenhum conteúdo cadastrado para o fonema {fonema} / nível {nivel}. "
                "Verifique se Dados/fonema_{fonema}/Palavras.docx existe e se "
                "extrator_conteudo.py já foi executado."
            ),
        )

    texto_gerado = formatar_para_nivel(item, nivel)

    if prova:
        instrucao = "🚨 MODO PROVA: Obtenha 90% ou mais para saltar de nível!"
    elif nivel == 1:
        instrucao = f"Lição do Fonema {fonema} — Atividade {passo + 1} de 10: Repita o som com firmeza."
    elif nivel == 2:
        instrucao = f"Nível 2 — Atividade {passo + 1} de 10: Fale a palavra unindo as partes devagar."
    elif nivel == 3:
        instrucao = f"Nível 3 — Atividade {passo + 1} de 10: Fale com clareza e ritmo natural."
    else:
        instrucao = f"Nível 4 — Atividade {passo + 1} de 10: Mantenha o fôlego até o fim."

    return {
        "usuario_id": usuario_id,
        "fonema": fonema,
        "nivel": nivel,
        "passo_licao": passo,
        "total_passos": 10,
        "modo_prova": prova,
        "texto_gerado": texto_gerado,
        "instrucao": instrucao,
    }


@app.post("/treino/enviar-audio")
async def enviar_audio_treino(
    usuario_id: int = Form(...),
    texto_esperado: str = Form(...),
    modo_prova: bool = Form(False),
    arquivo: UploadFile = File(...),
):
    if usuario_id not in USUARIOS_DB:
        USUARIOS_DB[usuario_id] = _usuario_padrao(f"Aluno {usuario_id}")

    usuario = USUARIOS_DB[usuario_id]

    resultado_mms = await chamar_mms_gop(texto_esperado=texto_esperado, arquivo=arquivo)
    score_geral = resultado_mms.get("score_geral", 0.0)
    detalhes_fonemas = resultado_mms.get("detalhes_fonemas", [])

    decisao = engine_adaptativa.processar_tentativa_licao(usuario, score_geral, modo_prova=modo_prova)

    fonemas_com_erro = [f["caractere"] for f in detalhes_fonemas if f["status"] != "correto"]
    resumo_erros = ", ".join(fonemas_com_erro) if fonemas_com_erro else "Nenhum"

    prompt_feedback = f"""
Você é o Fonoaudiólogo do LipAI.
Exercício da lição: "{texto_esperado}". Acurácia: {score_geral}%.
Sons com desvio: [{resumo_erros}].
Dê um feedback curto e positivo (1 frase) encorajando a criança.
"""
    feedback = (await run_in_threadpool(modelo_llm.invoke, prompt_feedback)).strip()

    return {
        "usuario_id": usuario_id,
        "analise_acustica": resultado_mms,
        "feedback_clinico": feedback,
        "decisao_adaptativa": decisao,
    }


@app.get("/")
def home():
    return {
        "status": "LipAI Core Lessons 4.0 Online",
        "banco_conteudo_carregado": len(BANCO_CONTEUDO) > 0,
        "letras_disponiveis": sorted(BANCO_CONTEUDO.keys()),
    }
