"""
LipAI LLM API — Lições de Fonemas + Portão de IA para o back-end.

Dois grupos de rotas:

1. /treino/*  → fluxo com estado NA MEMÓRIA desta API (USUARIOS_DB).
   É o que a página de teste `index.html` usa. NÃO foi alterado.

2. /ia/*      → portão STATELESS usado pelo back-end (.NET).
   Esta API não guarda progresso: recebe o `estado_adaptativo` (JSON) do
   back-end, processa e devolve o estado atualizado. Quem persiste é o back-end.
   Também é aqui que o MMS (motor acústico, porta 8000) é orquestrado.
"""

import json
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query

load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from adaptive_engine import engine_adaptativa

try:
    import pyphen

    _HIFENIZADOR = pyphen.Pyphen(lang="pt_BR")
except ImportError:
    _HIFENIZADOR = None

app = FastAPI(title="LipAI LLM API - Lições de Fonemas", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # com allow_origins=["*"] o padrão da spec exige credentials desligado
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #
PASTA_CHROMA = "./chroma_db"
CAMINHO_BANCO_CONTEUDO = "./banco_conteudo.json"

MMS_API_URL = os.getenv("MMS_API_URL", "http://127.0.0.1:8000/avaliar-fala-alinhada/")
MMS_TIMEOUT = float(os.getenv("MMS_TIMEOUT", "120"))

LLM_ENABLED = os.getenv("LLM_ENABLED", "1") not in ("0", "false", "False")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=PASTA_CHROMA, embedding_function=embeddings)

# O LLM é opcional: se o Ollama não estiver de pé, a API cai no feedback
# determinístico. A construção é preguiçosa para o import nunca quebrar.
_modelo_llm = None


def _obter_llm():
    global _modelo_llm
    if not LLM_ENABLED:
        return None
    if _modelo_llm is None:
        from langchain_ollama import OllamaLLM

        _modelo_llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
            num_predict=60,
        )
    return _modelo_llm


# --------------------------------------------------------------------------- #
# Banco de conteúdo (gerado por extrator_conteudo.py a partir dos .docx)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Estado adaptativo (compartilhado entre /treino/* e /ia/*)
# --------------------------------------------------------------------------- #
def _estado_adaptativo_default() -> dict:
    return {
        "fonema_atual": "A",
        "nivel_atual": 1,
        "passo_licao": 0,
        "scores_licao": [],
        "historico_itens": {},
        "historico_tentativas": [],
    }


def _usuario_padrao(nome: str = "Aluno") -> dict:
    return {"nome": nome, **_estado_adaptativo_default()}


def _carregar_estado(estado_json: str | None) -> dict:
    base = _estado_adaptativo_default()
    if not estado_json:
        return base
    try:
        recebido = json.loads(estado_json)
        if isinstance(recebido, dict):
            base.update(recebido)
    except (json.JSONDecodeError, TypeError):
        pass
    return base


USUARIOS_DB: dict[int, dict] = {1: _usuario_padrao("Criança Teste")}


# --------------------------------------------------------------------------- #
# Integração com o MMS (motor acústico)
# --------------------------------------------------------------------------- #
async def chamar_mms_gop(texto_esperado: str, filename: str, conteudo_audio: bytes) -> dict:
    try:
        async with httpx.AsyncClient() as cliente:
            files = {"arquivo": (filename or "audio.wav", conteudo_audio, "audio/wav")}
            data = {"texto_esperado": texto_esperado}
            res = await cliente.post(MMS_API_URL, data=data, files=files, timeout=MMS_TIMEOUT)
        if res.status_code != 200:
            raise HTTPException(status_code=502, detail=f"MMS respondeu {res.status_code}.")
        return res.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erro de conexão com o MMS: {e}")


# --------------------------------------------------------------------------- #
# Feedback do "fonoaudiólogo" (LLM com fallback determinístico)
# --------------------------------------------------------------------------- #
def _feedback_deterministico(score: float, fonemas_com_erro: list[str]) -> str:
    if score >= 90:
        return "Excelente! Sua pronúncia ficou muito clara. Continue assim! 🌟"
    if score >= 70:
        if fonemas_com_erro:
            return (
                f"Muito bem! Você acertou a maior parte. "
                f"Só reforce o som de {', '.join(fonemas_com_erro[:3])} da próxima vez."
            )
        return "Muito bem! Pronúncia dentro do esperado. Vamos para a próxima! 👏"
    if score >= 45:
        alvo = f" nos sons de {', '.join(fonemas_com_erro[:3])}" if fonemas_com_erro else ""
        return (
            f"Você está no caminho certo! Tente falar mais devagar e com a boca "
            f"bem aberta{alvo}. Respire fundo e tente de novo. 💪"
        )
    return (
        "Não desanime! Olhe o movimento dos lábios com calma, respire e repita "
        "o som devagar. Cada tentativa te deixa mais perto. 🙂"
    )


async def gerar_feedback_clinico(texto_esperado: str, score: float, detalhes: list[dict]) -> dict:
    fonemas_com_erro = [
        d["caractere"] for d in detalhes if d.get("status") and d["status"] != "correto"
    ]
    fallback = _feedback_deterministico(score, fonemas_com_erro)

    llm = _obter_llm()
    if llm is None:
        return {"feedback": fallback, "origem": "deterministico"}

    resumo_erros = ", ".join(fonemas_com_erro) if fonemas_com_erro else "Nenhum"
    prompt = (
        "Você é o Fonoaudiólogo do LipAI, atendendo uma criança.\n"
        f'Exercício: "{texto_esperado}". Acurácia da pronúncia: {score}%.\n'
        f"Sons com desvio: [{resumo_erros}].\n"
        "Dê um feedback curto (1 frase), positivo e encorajador, em português."
    )
    try:
        texto = (await run_in_threadpool(llm.invoke, prompt)).strip()
        return {"feedback": texto or fallback, "origem": "ollama"}
    except Exception as e:
        print(f"[app_llm] Ollama indisponível ({e}); usando feedback determinístico.")
        return {"feedback": fallback, "origem": "deterministico"}


# --------------------------------------------------------------------------- #
# Dica de fonema (RAG)
# --------------------------------------------------------------------------- #
async def buscar_dica_fonema(fonema: str) -> str:
    fonema = (fonema or "A").upper()
    termo = f"posição dos lábios e da língua, passagem do ar, treino visual do fonema {fonema}"
    resultados = await run_in_threadpool(
        db.similarity_search, termo, k=2, filter={"fonema": fonema}
    )
    if resultados:
        return "\n\n".join(d.page_content for d in resultados)
    return (
        f"Posicione os lábios e a língua de forma relaxada para emitir o som do "
        f"fonema {fonema} com clareza."
    )


def _montar_exercicio(estado: dict, modo_prova: bool) -> dict | None:
    fonema = estado["fonema_atual"]
    nivel = estado["nivel_atual"]
    passo = estado.get("passo_licao", 0)

    item = engine_adaptativa.proximo_item(estado, BANCO_CONTEUDO, modo_prova=modo_prova)
    if item is None:
        return None

    texto_gerado = formatar_para_nivel(item, nivel)

    if modo_prova:
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
        "fonema": fonema,
        "nivel": nivel,
        "passo_licao": passo,
        "total_passos": 10,
        "modo_prova": modo_prova,
        "texto_gerado": texto_gerado,
        "instrucao": instrucao,
    }


# =========================================================================== #
# GRUPO 1 — /treino/*  (estado na memória, usado pelo index.html de teste)
# =========================================================================== #
@app.post("/treino/selecionar-nivel/{usuario_id}")
def selecionar_nivel_manual(
    usuario_id: int, nivel: int = Query(..., ge=1, le=4), fonema: str = Query("A")
):
    usuario = USUARIOS_DB.setdefault(usuario_id, _usuario_padrao(f"Aluno {usuario_id}"))
    usuario["nivel_atual"] = nivel
    usuario["fonema_atual"] = fonema.upper()
    usuario["passo_licao"] = 0
    usuario["scores_licao"] = []
    return {"status": "sucesso", "usuario": usuario}


@app.get("/treino/dica-fonema/{usuario_id}")
async def obter_dica_fonema(usuario_id: int):
    usuario = USUARIOS_DB.get(usuario_id, {"fonema_atual": "A"})
    fonema = usuario["fonema_atual"]
    return {"fonema": fonema, "dica": await buscar_dica_fonema(fonema)}


@app.get("/treino/proximo-exercicio/{usuario_id}")
async def obter_proximo_exercicio(usuario_id: int, prova: bool = False):
    usuario = USUARIOS_DB.setdefault(usuario_id, _usuario_padrao(f"Aluno {usuario_id}"))
    exercicio = _montar_exercicio(usuario, modo_prova=prova)
    if exercicio is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Nenhum conteúdo para o fonema {usuario['fonema_atual']} / "
                f"nível {usuario['nivel_atual']}. Rode extrator_conteudo.py."
            ),
        )
    return {"usuario_id": usuario_id, **exercicio}


@app.post("/treino/enviar-audio")
async def enviar_audio_treino(
    usuario_id: int = Form(...),
    texto_esperado: str = Form(...),
    modo_prova: bool = Form(False),
    arquivo: UploadFile = File(...),
):
    usuario = USUARIOS_DB.setdefault(usuario_id, _usuario_padrao(f"Aluno {usuario_id}"))

    conteudo = await arquivo.read()
    resultado_mms = await chamar_mms_gop(texto_esperado, arquivo.filename, conteudo)
    score_geral = resultado_mms.get("score_geral", 0.0)
    detalhes = resultado_mms.get("detalhes_fonemas", [])

    decisao = engine_adaptativa.processar_tentativa_licao(usuario, score_geral, modo_prova=modo_prova)
    feedback = await gerar_feedback_clinico(texto_esperado, score_geral, detalhes)

    return {
        "usuario_id": usuario_id,
        "analise_acustica": resultado_mms,
        "feedback_clinico": feedback["feedback"],
        "feedback_origem": feedback["origem"],
        "decisao_adaptativa": decisao,
    }


# =========================================================================== #
# GRUPO 2 — /ia/*  (stateless, usado pelo back-end .NET)
# =========================================================================== #
@app.get("/ia/health")
async def ia_health():
    mms_ok = False
    try:
        async with httpx.AsyncClient() as c:
            base = MMS_API_URL.rsplit("/avaliar-fala-alinhada/", 1)[0]
            r = await c.get(base + "/", timeout=5)
            mms_ok = r.status_code == 200
    except Exception:
        mms_ok = False

    ollama_ok = False
    if LLM_ENABLED:
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
                ollama_ok = r.status_code == 200
        except Exception:
            ollama_ok = False

    return {
        "status": "online",
        "mms_ok": mms_ok,
        "ollama_ok": ollama_ok,
        "llm_habilitado": LLM_ENABLED,
        "feedback_modo": "ollama" if ollama_ok else "deterministico",
        "banco_conteudo_carregado": len(BANCO_CONTEUDO) > 0,
        "fonemas_disponiveis": sorted(BANCO_CONTEUDO.keys()),
    }


@app.get("/ia/dica-fonema")
async def ia_dica_fonema(fonema: str = Query("A")):
    return {"fonema": fonema.upper(), "dica": await buscar_dica_fonema(fonema)}


@app.get("/ia/proximo-exercicio")
async def ia_proximo_exercicio(
    estado_adaptativo: str = Query("", description="JSON do estado adaptativo do usuário"),
    modo_prova: bool = Query(False),
):
    estado = _carregar_estado(estado_adaptativo)
    exercicio = _montar_exercicio(estado, modo_prova=modo_prova)
    if exercicio is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Nenhum conteúdo para o fonema {estado['fonema_atual']} / "
                f"nível {estado['nivel_atual']}."
            ),
        )
    dica = await buscar_dica_fonema(estado["fonema_atual"])
    return {**exercicio, "dica": dica, "estado_adaptativo": estado}


@app.post("/ia/avaliar-fala")
async def ia_avaliar_fala(
    texto_esperado: str = Form(...),
    arquivo: UploadFile = File(...),
    estado_adaptativo: str = Form(""),
    modo_prova: bool = Form(False),
):
    estado = _carregar_estado(estado_adaptativo)

    conteudo = await arquivo.read()
    resultado_mms = await chamar_mms_gop(texto_esperado, arquivo.filename, conteudo)

    score_geral = float(resultado_mms.get("score_geral", 0.0))
    texto_falado = resultado_mms.get("texto_falado", "")
    detalhes = resultado_mms.get("detalhes_fonemas", [])

    decisao = engine_adaptativa.processar_tentativa_licao(estado, score_geral, modo_prova=modo_prova)
    feedback = await gerar_feedback_clinico(texto_esperado, score_geral, detalhes)

    return {
        "sucesso": True,
        "texto_esperado": texto_esperado,
        "texto_falado": texto_falado,
        "score_geral": score_geral,
        "detalhes_fonemas": detalhes,
        "feedback_clinico": feedback["feedback"],
        "feedback_origem": feedback["origem"],
        "decisao_adaptativa": decisao,
        "estado_adaptativo": estado,
    }


@app.get("/")
def home():
    return {
        "status": "LipAI Core Lessons 5.0 Online",
        "banco_conteudo_carregado": len(BANCO_CONTEUDO) > 0,
        "letras_disponiveis": sorted(BANCO_CONTEUDO.keys()),
    }
