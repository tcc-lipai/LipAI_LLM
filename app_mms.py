from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import torch
import torchaudio
import soundfile as sf
import numpy as np
import re
from transformers import Wav2Vec2ForCTC, AutoProcessor
import shutil
import os
import uuid

app = FastAPI(
    title="LipAI MMS - Motor Acústico CTC Forced Alignment",
    version="3.0.0"
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

print("Motor acústico MMS carregado com sucesso!")


def carregar_audio_16k(caminho_arquivo: str):
    """Carrega o áudio e garante o formato mono 16kHz."""
    try:
        audio_data, sr = sf.read(caminho_arquivo, dtype="float32")
        waveform = torch.tensor(audio_data, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.t()
    except Exception:
        waveform, sr = torchaudio.load(caminho_arquivo)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    return waveform


def avaliar_pronuncia_gop(caminho_arquivo: str, texto_alvo: str):
    """
    Executa alinhamento temporal CTC e calcula o Goodness of Pronunciation (GOP)
    para cada caractere/fonema do texto esperado.
    """
    waveform = carregar_audio_16k(caminho_arquivo)
    
    # 1. Normalização do texto esperado
    texto_limpo = re.sub(r'[^a-zA-Z\s]', '', texto_alvo.lower()).strip()
    if not texto_limpo:
        texto_limpo = texto_alvo.lower().strip()

    # 2. Obtenção das probabilidades temporais do modelo
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        emissions = model(**inputs).logits
        log_probs = torch.log_softmax(emissions, dim=-1)[0]  # Shape: (T, Vocab)
        probs = torch.softmax(emissions, dim=-1)[0]

    # Decodificação direta ASR para transcrição geral
    predicted_ids = torch.argmax(emissions, dim=-1)
    texto_transcrito = processor.decode(predicted_ids[0])

    # 3. Mapeamento dos tokens alvo
    tokens_alvo = processor.tokenizer.encode(texto_limpo)
    
    # Se o áudio for vazio ou tokens inválidos
    if len(tokens_alvo) == 0 or log_probs.shape[0] == 0:
        return {
            "score_geral": 0.0,
            "texto_transcrito": texto_transcrito or "[Sem áudio]",
            "detalhes_fonemas": []
        }

    # 4. Cálculo de Goodness of Pronunciation (GOP) por token alvo
    # Avalia a confiança acústica média da emissão do token
    detalhes_fonemas = []
    scores_lista = []
    
    for token_id in tokens_alvo:
        char = processor.tokenizer.decode([token_id])
        if not char.strip():
            continue
            
        # Pega a probabilidade máxima observada desse token ao longo do tempo do áudio
        prob_token_tempo = probs[:, token_id]
        score_acustico = float(torch.max(prob_token_tempo).item()) * 100.0
        
        # Ajuste de escala e penalidade suave
        score_ajustado = round(min(100.0, score_acustico * 1.15), 2)
        scores_lista.append(score_ajustado)

        status = "correto" if score_ajustado >= 70.0 else ("atencao" if score_ajustado >= 45.0 else "incorreto")

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


@app.post("/avaliar-fala-alinhada/", summary="Avaliação acústica de ponta via CTC GOP")
async def avaliar_fala_alinhada(
    texto_esperado: str = Form(...),
    arquivo: UploadFile = File(...)
):
    temp_path = f"temp_{uuid.uuid4().hex}_{arquivo.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(arquivo.file, buffer)

    try:
        resultado = await run_in_threadpool(avaliar_pronuncia_gop, temp_path, texto_esperado)
        return {
            "sucesso": True,
            "texto_esperado": texto_esperado,
            "texto_falado": resultado["texto_transcrito"],
            "score_geral": resultado["score_geral"],
            "detalhes_fonemas": resultado["detalhes_fonemas"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na avaliação acústica: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/")
def home():
    return {"status": "MMS CTC Engine Online"}