import os
import re
import json

import docx2txt

PASTA_DADOS = "./Dados"
SAIDA_JSON = "./banco_conteudo.json"

MARCADORES = [
    ("nivel1", r"N[íi]vel\s*1\s*Vogais e Sons"),
    ("nivel2", r"Palavras\s*N[íi]vel\s*2"),
    ("nivel3_palavras", r"Palavras\s*N[íi]vel\s*3"),
    ("nivel3_frases", r"Frases\s*e\s*Express(?:õ|o)es\s*Curtas"),
    ("nivel4", r"Palavras\s*N[íi]vel\s*4"),
    ("vocabulario", r"Vocabul[áa]rio\s*Avan[çc]ado"),
]

CABECALHOS = {
    "nivel1a": r"^N[íi]vel\s*1\s*Vogais e Sons\s*\([^)]*\)\s*:\s*",
    "nivel1b": r"^Encontros Vocálicos e Ditongos:?\s*",
    "nivel2": r"^Palavras\s*N[íi]vel\s*2\s*\([^)]*\)\s*",
    "nivel3_palavras": r"^Palavras\s*N[íi]vel\s*3\s*\([^)]*\)\s*",
    "nivel3_frases": r"^Frases\s*e\s*Express(?:õ|o)es\s*Curtas\s*",
    "nivel4": r"^Palavras\s*N[íi]vel\s*4\s*e\s*Frases\s*Complexas\s*",
    "vocabulario": r"^Vocabul[áa]rio\s*Avan[çc]ado\s*e\s*Conceitos\s*com\s*[A-Za-zÀ-ú]\s*",
}


def _remover_cabecalho(texto: str, chave: str) -> str:
    return re.sub(CABECALHOS[chave], "", texto.strip(), flags=re.IGNORECASE | re.DOTALL).strip()


def limpar_lista(texto: str, chave: str) -> list:
    """Para blocos separados por vírgula (sílabas e palavras soltas)."""
    texto = _remover_cabecalho(texto, chave)
    itens = [i.strip(" .") for i in re.split(r",", texto)]
    return [i for i in itens if i]


def limpar_frases(texto: str, chave: str) -> list:
    """Para blocos de frases/sentenças separadas por ponto final."""
    texto = _remover_cabecalho(texto, chave)
    frases = [f.strip(" .") for f in re.split(r"\.\s+", texto) if f.strip(" .")]
    return frases


def limpar_vocabulario(texto: str, chave: str) -> list:
    """Para a lista final de vocabulário avançado (palavras separadas por espaço)."""
    texto = _remover_cabecalho(texto, chave)
    return [p.strip(" .") for p in texto.split() if p.strip(" .")]


def extrair_secoes(texto_completo: str) -> dict:
    texto_completo = re.sub(r"\s+", " ", texto_completo).strip()

    posicoes = []
    for nome, padrao in MARCADORES:
        m = re.search(padrao, texto_completo, re.IGNORECASE)
        if m:
            posicoes.append((m.start(), nome))
    posicoes.sort()

    blocos = {}
    for i, (inicio, nome) in enumerate(posicoes):
        fim = posicoes[i + 1][0] if i + 1 < len(posicoes) else len(texto_completo)
        blocos[nome] = texto_completo[inicio:fim].strip()

    secoes = {
        "nivel1": [], "nivel2": [], "nivel3_palavras": [],
        "nivel3_frases": [], "nivel4": [], "vocabulario": [],
    }

    if "nivel1" in blocos:
        bloco = blocos["nivel1"]
        partes = re.split(r"(?=Encontros Vocálicos e Ditongos:?)", bloco, flags=re.IGNORECASE)
        if len(partes) >= 1:
            secoes["nivel1"].extend(limpar_lista(partes[0], "nivel1a"))
        if len(partes) >= 2:
            secoes["nivel1"].extend(limpar_lista(partes[1], "nivel1b"))
        secoes["nivel1"] = list(dict.fromkeys(secoes["nivel1"]))  # remove duplicados, mantém ordem

    if "nivel2" in blocos:
        secoes["nivel2"] = limpar_lista(blocos["nivel2"], "nivel2")
    if "nivel3_palavras" in blocos:
        secoes["nivel3_palavras"] = limpar_lista(blocos["nivel3_palavras"], "nivel3_palavras")
    if "nivel3_frases" in blocos:
        secoes["nivel3_frases"] = limpar_frases(blocos["nivel3_frases"], "nivel3_frases")
    if "nivel4" in blocos:
        secoes["nivel4"] = limpar_frases(blocos["nivel4"], "nivel4")
    if "vocabulario" in blocos:
        secoes["vocabulario"] = limpar_vocabulario(blocos["vocabulario"], "vocabulario")

    return secoes


def construir_banco():
    if not os.path.exists(PASTA_DADOS):
        print(f" Pasta {PASTA_DADOS} não encontrada.")
        return

    banco = {}
    pastas = sorted(os.listdir(PASTA_DADOS))

    for pasta in pastas:
        m = re.match(r"fonema_([A-Za-zÀ-ú])", pasta, re.IGNORECASE)
        if not m:
            continue
        letra = m.group(1).upper()
        caminho_palavras = os.path.join(PASTA_DADOS, pasta, "Palavras.docx")

        if not os.path.exists(caminho_palavras):
            print(f"  {pasta}: Palavras.docx não encontrado, pulando.")
            continue

        texto = docx2txt.process(caminho_palavras)
        secoes = extrair_secoes(texto)

        total_itens = sum(len(v) for v in secoes.values())
        if total_itens == 0:
            print(f"  {pasta}: NENHUM item extraído — confira os títulos das seções no Word.")
        else:
            print(
                f" {pasta}: nível1={len(secoes['nivel1'])} nível2={len(secoes['nivel2'])} "
                f"nível3_palavras={len(secoes['nivel3_palavras'])} nível3_frases={len(secoes['nivel3_frases'])} "
                f"nível4={len(secoes['nivel4'])} vocabulário={len(secoes['vocabulario'])}"
            )

        banco[letra] = secoes

    with open(SAIDA_JSON, "w", encoding="utf-8") as f:
        json.dump(banco, f, ensure_ascii=False, indent=2)

    print(f"\nBanco de conteúdo salvo em {SAIDA_JSON} ({len(banco)} letras processadas).")
    print("   Rode este script de novo sempre que editar os Palavras.docx.")


if __name__ == "__main__":
    construir_banco()