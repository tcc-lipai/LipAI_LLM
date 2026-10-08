import math
import os
import re
import tempfile
import uuid
from difflib import SequenceMatcher

import av
import numpy as np
import phonetics
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from transformers import Wav2Vec2ForCTC, AutoProcessor

app = FastAPI(
    title="LipAI MMS - Motor Acústico CTC Forced Alignment",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = "facebook/mms-1b-all"
print("Carregando o modelo MMS para alinhamento CTC...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

processor.tokenizer.set_target_lang("por")
model.load_adapter("por")
model.eval()

BLANK_ID = processor.tokenizer.pad_token_id

print("Motor acústico MMS carregado com sucesso!")

LIMIAR_CORRETO = 70.0
LIMIAR_ATENCAO = 45.0

TAMANHO_MAXIMO_AUDIO_BYTES = 15 * 1024 * 1024


def _carregar_com_pyav(caminho_arquivo: str):
    container = av.open(caminho_arquivo)
    stream = container.streams.audio[0]
    taxa_original = stream.rate
    resampler = av.AudioResampler(format="flt", layout="mono", rate=taxa_original)

    pedacos = []
    for frame in container.decode(stream):
        for frame_resampleado in resampler.resample(frame):
            pedacos.append(frame_resampleado.to_ndarray())
    container.close()

    if not pedacos:
        raise ValueError("Não foi possível decodificar nenhum quadro de áudio do arquivo.")

    audio = np.concatenate(pedacos, axis=1).flatten()
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0), taxa_original


def carregar_audio_16k(caminho_arquivo: str):
    try:
        audio_data, sr = sf.read(caminho_arquivo, dtype="float32")
        waveform = torch.tensor(audio_data, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.t()
    except Exception:
        try:
            waveform, sr = torchaudio.load(caminho_arquivo)
        except Exception:
            waveform, sr = _carregar_com_pyav(caminho_arquivo)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    return waveform


def _normalizar_texto_alvo(texto_alvo: str) -> str:
    # O tokenizer do adaptador "por" tem tokens próprios para acentos/cedilha
    # (á, â, ã, ç, é, ê, í, ó, ô, õ, ú, ...) — são fonemas diferentes das
    # letras simples (ex.: "ç" soa /s/, não /k/). Remover esses caracteres
    # antes do forced-align faz o alinhamento comparar o áudio com a
    # ortografia errada, derrubando a nota mesmo com pronúncia correta.
    # Por isso aqui só tiramos o que o tokenizer realmente não reconhece
    # (números, pontuação), sem mexer nos acentos.
    return re.sub(r"[^a-zàáâãçéêëíòóôõúü\s]", "", texto_alvo.lower()).strip()


def _normalizar_palavra_fonetica_br(palavra: str) -> str:
    """
    Aplica pra uma palavra as mudanças de som regulares e sistemáticas da
    fala natural do português brasileiro, que a ortografia não mostra:

    - vogal átona final: "o" -> "u" (ex.: "gato" soa "gatu")
    - vogal átona final: "e" -> "i" (ex.: "leite" soa "leiti")
    - "l" no final da palavra -> "u", vocalização do L (ex.: "ágil" soa "águiu")
    - "r" no final da palavra some/aspira (ex.: "agir" soa "agi")
    - "t"/"d" antes de "i" palatalizam pra "tch"/"dj" (ex.: "tia" soa "tchia",
      e "leiti" -- já convertido acima -- vira "leitchi")
    """
    if not palavra:
        return palavra

    if palavra.endswith("o"):
        palavra = palavra[:-1] + "u"
    elif palavra.endswith("e"):
        palavra = palavra[:-1] + "i"

    if palavra.endswith("l"):
        palavra = palavra[:-1] + "u"

    if palavra.endswith("r"):
        palavra = palavra[:-1]

    palavra = re.sub(r"t(?=i)", "tch", palavra)
    palavra = re.sub(r"d(?=i)", "dj", palavra)

    return palavra


def _normalizar_fonetica_br(texto: str) -> str:
    return " ".join(_normalizar_palavra_fonetica_br(p) for p in texto.split())


def calcular_score_texto(ouvido: str, alvo: str) -> float:
    """
    Reproduz o método de pontuação usado no teste de comparação no Colab:
    mistura similaridade textual (SequenceMatcher) com similaridade
    fonética (Metaphone), comparando a transcrição livre INTEIRA contra o
    texto esperado INTEIRO. Diferente do GOP por caractere (que trava a
    nota se um só fonema tiver confiança baixa), essa métrica é mais
    tolerante -- é mais sobre "o que saiu bate com o esperado?" do que
    "o modelo tem certeza de cada som esperado, na ordem certa?".

    Antes de comparar, os dois lados passam por _normalizar_fonetica_br --
    sem isso, pronúncia 100% natural e correta de brasileiro (tipo "ágil"
    soando "águiu") perderia nota só por não bater com a grafia literal.
    """
    if not ouvido:
        return 0.0
    ouvido = _normalizar_fonetica_br(ouvido.lower().strip().replace(".", ""))
    alvo = _normalizar_fonetica_br(alvo.lower().strip())
    sim_escrita = SequenceMatcher(None, ouvido, alvo).ratio() * 100
    sim_fonetica = (
        SequenceMatcher(None, phonetics.metaphone(ouvido), phonetics.metaphone(alvo)).ratio()
        * 100
    )
    return (sim_escrita * 0.4) + (sim_fonetica * 0.6)


def avaliar_pronuncia_gop(caminho_arquivo: str, texto_alvo: str):
    waveform = carregar_audio_16k(caminho_arquivo)

    texto_limpo = _normalizar_texto_alvo(texto_alvo)

    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        emissions = model(**inputs).logits
        log_probs = torch.log_softmax(emissions, dim=-1)

    predicted_ids = torch.argmax(emissions, dim=-1)
    texto_transcrito = processor.decode(predicted_ids[0])

    tokens_alvo = processor.tokenizer.encode(texto_limpo) if texto_limpo else []

    if len(tokens_alvo) == 0 or log_probs.shape[1] == 0:
        return {
            "score_geral": 0.0,
            "texto_transcrito": texto_transcrito or "[Sem áudio]",
            "detalhes_fonemas": []
        }

    targets = torch.tensor([tokens_alvo], dtype=torch.int32)
    aligned_tokens, aligned_scores = torchaudio.functional.forced_align(
        log_probs, targets, blank=BLANK_ID
    )
    spans = torchaudio.functional.merge_tokens(aligned_tokens[0], aligned_scores[0], blank=BLANK_ID)

    detalhes_fonemas = []
    scores_lista = []

    for posicao, span in enumerate(spans):
        if posicao >= len(texto_limpo):
            break
        char = texto_limpo[posicao]
        if not char.strip():
            continue

        score_ajustado = round(min(100.0, math.exp(span.score) * 100.0), 2)
        scores_lista.append(score_ajustado)

        status = (
            "correto" if score_ajustado >= LIMIAR_CORRETO
            else "atencao" if score_ajustado >= LIMIAR_ATENCAO
            else "incorreto"
        )

        detalhes_fonemas.append({
            "caractere": char.upper(),
            "score": score_ajustado,
            "status": status
        })

    # Nota geral no estilo do teste do Colab (texto transcrito x texto
    # esperado INTEIROS), em vez da média das confianças por caractere do
    # forced-align. O detalhamento por fonema abaixo continua vindo do
    # forced-align, só a nota geral mudou de método.
    score_geral = round(calcular_score_texto(texto_transcrito, texto_alvo), 2)

    return {
        "score_geral": score_geral,
        "texto_transcrito": texto_transcrito,
        "detalhes_fonemas": detalhes_fonemas
    }


async def _salvar_audio_temporario(arquivo: UploadFile) -> str:
    extensao = os.path.splitext(arquivo.filename or "")[1]
    if not re.fullmatch(r"\.[A-Za-z0-9]{1,5}", extensao):
        extensao = ".wav"

    caminho = os.path.join(tempfile.gettempdir(), f"lipai_mms_{uuid.uuid4().hex}{extensao}")

    tamanho_total = 0
    with open(caminho, "wb") as buffer:
        while True:
            pedaco = await arquivo.read(1024 * 1024)
            if not pedaco:
                break
            tamanho_total += len(pedaco)
            if tamanho_total > TAMANHO_MAXIMO_AUDIO_BYTES:
                buffer.close()
                os.remove(caminho)
                raise HTTPException(
                    status_code=413,
                    detail=f"Áudio excede o tamanho máximo permitido ({TAMANHO_MAXIMO_AUDIO_BYTES // (1024 * 1024)} MB).",
                )
            buffer.write(pedaco)

    return caminho


@app.post("/avaliar-fala-alinhada/", summary="Avaliação acústica de ponta via CTC GOP")
async def avaliar_fala_alinhada(
    texto_esperado: str = Form(...),
    arquivo: UploadFile = File(...)
):
    temp_path = await _salvar_audio_temporario(arquivo)

    try:
        resultado = await run_in_threadpool(avaliar_pronuncia_gop, temp_path, texto_esperado)
        return {
            "sucesso": True,
            "texto_esperado": texto_esperado,
            "texto_falado": resultado["texto_transcrito"],
            "score_geral": resultado["score_geral"],
            "detalhes_fonemas": resultado["detalhes_fonemas"]
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[app_mms] Erro na avaliação acústica: {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao avaliar o áudio.")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/")
def home():
    return {"status": "MMS CTC Engine Online"}
