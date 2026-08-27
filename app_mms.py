import math
import os
import re
import tempfile
import unicodedata
import uuid

import av
import numpy as np
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
    texto_sem_acento = unicodedata.normalize("NFKD", texto_alvo)
    texto_sem_acento = "".join(c for c in texto_sem_acento if not unicodedata.combining(c))
    return re.sub(r"[^a-zA-Z\s]", "", texto_sem_acento.lower()).strip()


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

    score_geral = round(float(np.mean(scores_lista)), 2) if scores_lista else 0.0

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
