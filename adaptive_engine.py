import random

TRILHA_ALFABETO = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
]

TOTAL_ATIVIDADES_LICAO = 10

# --- SISTEMA DE CAIXINHAS (Leitner) ---
# 0 = novo, 1 = fraco, 2 = aprendendo, 3 = bom, 4 = dominado
CAIXA_NOVO = 0
CAIXA_FRACO = 1
CAIXA_APRENDENDO = 2
CAIXA_BOM = 3
CAIXA_DOMINADO = 4
CAIXA_MAXIMA = CAIXA_DOMINADO

# Peso de sorteio de cada caixinha: quanto maior, mais vezes aquele item
# aparece. Itens fracos aparecem bem mais que itens dominados.
PESOS_CAIXA = {
    CAIXA_NOVO: 3.0,
    CAIXA_FRACO: 6.0,
    CAIXA_APRENDENDO: 3.0,
    CAIXA_BOM: 1.5,
    CAIXA_DOMINADO: 0.5,
}

# Score mínimo (na atividade individual) pra considerar que o item foi
# "acertado" e pode subir de caixinha.
LIMIAR_ACERTO_ITEM = 70.0

# Quantos itens recentes evitamos repetir de novo em seguida (se o pool
# for grande o suficiente pra isso valer a pena).
JANELA_ANTI_REPETICAO = 2


def _intercalar(lista_a: list, lista_b: list) -> list:
    """Intercala duas listas (usado no Nível 3: palavras + frases curtas)."""
    resultado = []
    i = j = 0
    while i < len(lista_a) or j < len(lista_b):
        if i < len(lista_a):
            resultado.append(lista_a[i])
            i += 1
        if j < len(lista_b):
            resultado.append(lista_b[j])
            j += 1
    return resultado


def obter_pool_conteudo(banco_conteudo: dict, fonema: str, nivel: int) -> list:
    """Retorna a lista determinística de itens (na ordem do Palavras.docx)
    para aquele fonema + nível. Nunca inventa conteúdo: só usa o que está
    no banco_conteudo.json gerado pelo extrator_conteudo.py."""
    dados_letra = banco_conteudo.get(fonema, {})

    if nivel == 1:
        return dados_letra.get("nivel1", [])
    if nivel == 2:
        return dados_letra.get("nivel2", [])
    if nivel == 3:
        palavras = dados_letra.get("nivel3_palavras", [])
        frases = dados_letra.get("nivel3_frases", [])
        return _intercalar(palavras, frases)
    # nível 4
    return dados_letra.get("nivel4", []) + dados_letra.get("vocabulario", [])


def _chave_item(fonema: str, nivel: int, item: str) -> str:
    return f"{fonema}_{nivel}_{item}"


def _obter_registro_item(usuario: dict, chave: str) -> dict:
    """Retorna (criando se preciso) o registro de histórico daquele item
    específico: tentativas, acertos e caixinha atual."""
    historico = usuario.setdefault("historico_itens", {})
    return historico.setdefault(
        chave, {"tentativas": 0, "acertos": 0, "caixa": CAIXA_NOVO}
    )


class AdaptiveEngine:
    def __init__(self):
        self.dificuldade_niveis = {1: -1.5, 2: -0.5, 3: 0.8, 4: 2.0}

    def proximo_item(self, usuario: dict, banco_conteudo: dict, modo_prova: bool = False):
        """Escolhe o próximo som/palavra/frase de forma ADAPTATIVA: itens que
        a criança erra mais (caixinha baixa) aparecem com mais frequência,
        itens já dominados aparecem raramente. Nunca inventa conteúdo — só
        sorteia dentro do pool determinístico do banco_conteudo.json. Retorna
        None se não houver conteúdo cadastrado para aquele fonema."""
        fonema = usuario["fonema_atual"]
        nivel = usuario["nivel_atual"]
        pool = obter_pool_conteudo(banco_conteudo, fonema, nivel)

        if not pool:
            return None

        if modo_prova:
            # Na prova de salto, usa o item mais avançado do pool daquele nível.
            item = pool[-1]
            usuario["_ultimo_item_servido"] = item
            return item

        # Últimos itens servidos pra esse fonema+nível, pra evitar repetir
        # o mesmo item duas vezes seguidas quando o pool permite.
        historico_recente = usuario.setdefault("_recentes_por_pool", {})
        chave_pool = f"{fonema}_{nivel}"
        recentes = historico_recente.get(chave_pool, [])

        candidatos = pool
        if len(pool) > JANELA_ANTI_REPETICAO:
            candidatos = [item for item in pool if item not in recentes] or pool

        pesos = []
        for item in candidatos:
            registro = _obter_registro_item(usuario, _chave_item(fonema, nivel, item))
            pesos.append(PESOS_CAIXA[registro["caixa"]])

        item_escolhido = random.choices(candidatos, weights=pesos, k=1)[0]

        # Atualiza a janela de itens recentes.
        recentes = (recentes + [item_escolhido])[-JANELA_ANTI_REPETICAO:]
        historico_recente[chave_pool] = recentes

        # Guarda qual foi o item servido pra essa tentativa, pra
        # processar_tentativa_licao saber qual caixinha atualizar depois.
        usuario["_ultimo_item_servido"] = item_escolhido
        return item_escolhido

    def _atualizar_caixinha_item(self, usuario: dict, score_acerto: float):
        """Atualiza tentativas/acertos/caixinha do item que acabou de ser
        testado, com base no score da tentativa individual."""
        item = usuario.get("_ultimo_item_servido")
        if item is None:
            return  # nada pra atualizar (ex: chamada fora do fluxo normal)

        fonema = usuario["fonema_atual"]
        nivel = usuario["nivel_atual"]
        chave = _chave_item(fonema, nivel, item)
        registro = _obter_registro_item(usuario, chave)

        registro["tentativas"] += 1
        if score_acerto >= LIMIAR_ACERTO_ITEM:
            registro["acertos"] += 1
            registro["caixa"] = min(CAIXA_MAXIMA, registro["caixa"] + 1)
        else:
            # Errou: volta pra "fraco" (não zera direto pra "novo", pra não
            # perder o progresso todo — mas some com pelo menos 2 caixinhas).
            registro["caixa"] = max(CAIXA_NOVO, registro["caixa"] - 2)

    def processar_tentativa_licao(self, usuario: dict, score_acerto: float, modo_prova: bool = False):
        nivel = usuario["nivel_atual"]
        fonema = usuario["fonema_atual"]

        # 0. Atualiza a caixinha do item específico que acabou de ser testado
        # (tanto em modo lição quanto em modo prova).
        self._atualizar_caixinha_item(usuario, score_acerto)

        # 1. Modo Prova
        if modo_prova:
            if score_acerto >= 90.0 and nivel < 4:
                usuario["nivel_atual"] += 1
                usuario["fonema_atual"] = "A"
                usuario["passo_licao"] = 0
                usuario["scores_licao"] = []
                return {
                    "licao_concluida": True,
                    "sucesso_prova": True,
                    "mensagem": f"Aprovado com {score_acerto}%! Você pulou para o Nível {usuario['nivel_atual']}!",
                    "nivel_atual": usuario["nivel_atual"],
                    "fonema_atual": usuario["fonema_atual"],
                }
            return {
                "licao_concluida": True,
                "sucesso_prova": False,
                "mensagem": f"Pontuação: {score_acerto}%. A prova exige no mínimo 90% para saltar de nível.",
                "nivel_atual": usuario["nivel_atual"],
                "fonema_atual": usuario["fonema_atual"],
            }

        # 2. Registro do score da atividade atual
        usuario["scores_licao"].append(score_acerto)
        usuario["passo_licao"] += 1

        licao_concluida = usuario["passo_licao"] >= TOTAL_ATIVIDADES_LICAO
        media_final = None
        proficiente = False
        mensagem = ""

        # 3. Finalização da Lição de 10 Atividades
        if licao_concluida:
            media_final = round(sum(usuario["scores_licao"]) / len(usuario["scores_licao"]), 2)
            proficiente = media_final >= 75.0

            if proficiente:
                mensagem = f"🏆 Parabéns! Você concluiu a lição do Fonema {fonema} com média de {media_final}% e está PROFICIENTE!"
                idx_atual = TRILHA_ALFABETO.index(fonema) if fonema in TRILHA_ALFABETO else 0
                if idx_atual + 1 < len(TRILHA_ALFABETO):
                    usuario["fonema_atual"] = TRILHA_ALFABETO[idx_atual + 1]
                else:
                    if nivel < 4:
                        usuario["nivel_atual"] += 1
                        usuario["fonema_atual"] = "A"
            else:
                mensagem = f"⚠️ Sua média na lição do Fonema {fonema} foi de {media_final}%. É recomendado repetir a lição para fixar o ponto articulatório."

            usuario["passo_licao"] = 0
            usuario["scores_licao"] = []

        return {
            "licao_concluida": licao_concluida,
            "passo_atual": usuario["passo_licao"],
            "total_passos": TOTAL_ATIVIDADES_LICAO,
            "media_final": media_final,
            "proficiente": proficiente,
            "mensagem": mensagem,
            "nivel_atual": usuario["nivel_atual"],
            "fonema_atual": usuario["fonema_atual"],
        }


engine_adaptativa = AdaptiveEngine()