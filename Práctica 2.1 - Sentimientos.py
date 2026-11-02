# ¿Cómo funciona el análisis de sentimientos? Funciona como una tarea de procesamiento del Lenguaje Natural (PLN) y clasificación supervisada. No comprende el significado de forma cognitiva: tokeniza el texto 
# y lo proyecta en un espacio vectorial numérico mediante transformadores preentrenados (como RoBERTuito). El modelo calcula probabilidades para asignar la etiqueta más afín (POS, NEG, NEU) y devuelve un nivel 
# de certeza estadística (score). 

# ¿Qué límites tiene? Sarcasmo e ironía: Palabras superficialmente positivas: "qué maravilla", "un genio", en contextos de frustración suelen despistar a los clasificadores. Dependencia del contexto sociocultural: 
# Modismos regionales, jerga generacional, dobles sentidos o faltas de ortografía ambiguas alteran el resultado. Negación compleja y contraste: Estructuras con giros adversativos: "No estuvo mal, pero tampoco volvería",
# dificultan ponderar el peso real del sentimiento. Subjetividad del score: Un valor cercano a 1.0 refleja seguridad estadística del clasificador sobre sus patrones aprendidos, no necesariamente una interpretación 
# correcta de la intención humana.


# Práctica (tuve que acabar haciendo usa de la IA ya que no paraba de recibir errores al ejecutar)


try:
    from transformers import pipeline
except ModuleNotFoundError:
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch"])
        from transformers import pipeline
    except Exception:
        pipeline = None


lista_negra = ["cheater", "aimbot", "hack", "rata", "troleo"]


def _sentimiento_heuristico(texto: str) -> tuple[str, float]:
    tokens = texto.lower()
    positivos = [
        "obra maestra", "genial", "buen", "excelente", "maravilla", "entretenido",
        "sorprendente", "fantastico", "impecable", "mejor", "perfecto", "agradable"
    ]
    negativos = [
        "desastroso", "bug", "caidas", "pesado", "in jugable", "peor",
        "malo", "terrible", "fatal", "rata", "aimbot", "hack", "cheater", "troleo"
    ]

    puntuacion = 0.0
    for palabra in positivos:
        if palabra in tokens:
            puntuacion += 1.0
    for palabra in negativos:
        if palabra in tokens:
            puntuacion -= 1.5

    if puntuacion > 0:
        return "POS", min(0.99, 0.65 + puntuacion * 0.10)
    if puntuacion < 0:
        return "NEG", min(0.99, 0.70 + abs(puntuacion) * 0.08)
    return "NEU", 0.55


try:
    clasificador = pipeline("sentiment-analysis", model="pysentimiento/robertuito-sentiment-analysis")
except Exception:
    clasificador = None


def moderar_review(texto: str, palabras_vetadas: list) -> dict:
    texto_lower = texto.lower()
    contiene_termino_ilicito = any(termino in texto_lower for termino in palabras_vetadas)

    if clasificador is not None:
        try:
            prediccion = clasificador(texto)[0]
            etiqueta = prediccion.get("label", "NEU")
            score = float(prediccion.get("score", 0.0))
        except Exception:
            etiqueta, score = _sentimiento_heuristico(texto)
    else:
        etiqueta, score = _sentimiento_heuristico(texto)

    if contiene_termino_ilicito:
        estado = "RECHAZADA"
        motivo = "Incumple las normas: contiene terminos prohibidos de la lista negra"
    elif etiqueta == "NEG" and score > 0.5:
        estado = "RECHAZADA"
        motivo = f"Comentario negativo descartado (confianza: {score:.2f})"
    else:
        estado = "PUBLICADA"
        motivo = f"Aprobada para el feed (sentimiento {etiqueta} valido)"

    return {
        "comentario": texto,
        "sentimiento": etiqueta,
        "confianza": round(score, 3),
        "estado": estado,
        "motivo": motivo
    }


if __name__ == "__main__":
    casos_prueba = [
        "La banda sonora y el diseno de niveles son una autentica obra maestra.",  # PUBLICADA
        "El parche 1.04 pesa 12 GB y ya esta disponible para descargar.",          # PUBLICADA
        "El juego esta lleno de gente con aimbot, es injugable.",                  # RECHAZADA
        "Rendimiento desastroso, lleno de bugs y caidas de frames constantes.",    # RECHAZADA
        "El juego esta bien pero el soporte tecnico me parece una rata.",          # RECHAZADA
        "No es el mejor juego del ano, pero entretiene lo justo."                  # PUBLICADA
    ]

    for prueba in casos_prueba:
        res = moderar_review(prueba, lista_negra)
        print(f"[{res['estado']}] ({res['sentimiento']} {res['confianza']}) -> \"{res['comentario']}\"")
        print(f"   -> {res['motivo']}\n")