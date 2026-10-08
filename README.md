# LipAI — IA (motor acústico + lições de fonemas)

Dois serviços FastAPI:

| Serviço      | Porta | Papel                                                            |
| ------------ | ----- | --------------------------------------------------------------- |
| `app_mms`    | 8000  | Motor acústico (MMS-1b, CTC forced alignment). Dá a nota da fala. |
| `app_llm`    | 8001  | Portão da IA: RAG de dicas, motor adaptativo, feedback. Chama o `app_mms` por baixo. |

O **back-end (.NET) só conversa com o `app_llm` (8001)**. O `app_mms` fica atrás dele.

## Instalação

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> `torch`/`torchaudio` são as builds `+cpu`. Se o pip não achar, use:
> `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`

Gere os dados (lê os `.docx` de `Dados/`):

```bash
python extrator_conteudo.py   # -> banco_conteudo.json
python criar_rag.py           # -> chroma_db/  (índice das dicas)
```

## Rodando

Em terminais separados:

```bash
uvicorn app_mms:app --port 8000
uvicorn app_llm:app --port 8001
```

O `app_mms` baixa o modelo `facebook/mms-1b-all` do Hugging Face no primeiro start
(alguns minutos). Depois fica em cache.

## Configuração (`.env`, opcional)

Copie `.env.example` para `.env`. Tudo tem padrão sensato. O principal:

- `LLM_ENABLED=0` desliga o Ollama e usa o **feedback determinístico** (regras
  sobre a nota + fonemas com erro). É o modo que roda sem dependência externa.
- Com `LLM_ENABLED=1` (padrão) e o **Ollama** rodando (`ollama serve` +
  `ollama pull qwen2.5:3b`), o feedback vira texto gerado pelo LLM.

## Endpoints do portão (`/ia/*`, usados pelo back-end)

| Método | Rota                     | O que faz |
| ------ | ------------------------ | --------- |
| GET    | `/ia/health`             | Status do MMS, Ollama, RAG e banco de conteúdo |
| GET    | `/ia/dica-fonema?fonema=A` | Dica articulatória do fonema (RAG) |
| GET    | `/ia/proximo-exercicio`  | Próximo item da lição (recebe `estado_adaptativo` em JSON) |
| POST   | `/ia/avaliar-fala`       | Recebe áudio + `texto_esperado` + `estado_adaptativo`; devolve nota, fonemas, feedback e estado atualizado |

Esses endpoints são **stateless**: o progresso do aluno é mandado e devolvido no
campo `estado_adaptativo`; quem guarda é o back-end.

## Página de teste (`index.html`)

Abra direto no navegador. Usa as rotas `/treino/*` (que mantêm estado na memória
desta API) — serve só para testar o motor de lições isoladamente.
