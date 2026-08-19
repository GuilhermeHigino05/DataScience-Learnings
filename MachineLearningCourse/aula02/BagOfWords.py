#& =====================================================================
#& BAGOFWORDS.PY — de texto bruto para uma matriz numérica (Bag of Words).
#& ---------------------------------------------------------------------
#& O que mudou nesta versão:
#&   * `create_bow_vectorizer` agora devolve também os textos já
#&     pré-processados, para que o pré-processamento aconteça UMA única
#&     vez (antes ele rodava de novo dentro de `GetFeatures`).
#&   * `GetFeatures` devolve a matriz ESPARSA do scipy, em vez de chamar
#&     `.toarray()`. Bag of Words é ~99% zeros; o formato esparso guarda
#&     só os não-zeros, economizando MUITA memória e deixando cada produto
#&     escalar muito mais rápido.
#& =====================================================================
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

def preprocess_text(text):
    """
    Função para pré-processar texto:
    - Converte para minúsculas
    - Remove caracteres especiais e números
    - Remove stopwords
    """
    if isinstance(text, str):
        # Converter para minúsculas
        text = text.lower()
        
        # Remover caracteres especiais e números
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remover stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))

        words = text.split()
        words = [word for word in words if word not in stop_words]
        
        return " ".join(words)
    else:
        return ""

def create_bow_vectorizer(conjunto_textos):
    """
    Função para criar o vetorizador Bag of Words a partir do CSV.
    Devolve (vectorizer, textos_processados) para reutilizar o
    pré-processamento.
    """
    #& O `CountVectorizer` aprende o vocabulário (fit); depois, cada texto
    #& vira um vetor de contagens. O pré-processamento é feito aqui e
    #& devolvido junto, para que `GetFeatures` não precise refazê-lo.

    # Pré-processa cada texto UMA vez: minúsculas, sem pontuação/números
    # e sem stopwords. Antes, isso rodava duas vezes (aqui e no GetFeatures).
    conjunto_textos_processados = conjunto_textos.apply(preprocess_text)

    # Ajusta o vocabulário aos textos processados.
    vectorizer = CountVectorizer()
    vectorizer.fit(conjunto_textos_processados)

    # Devolve o vetorizador E os textos já processados (uma tupla).
    return vectorizer, conjunto_textos_processados

def GetFeatures(textos_preprocessados, vectorizer):
    """
    Transforma os textos (já pré-processados) em uma matriz Bag of Words.

    Retorna a matriz esparsa (sparse) do scipy, muito mais eficiente em
    memória e tempo do que o formato denso (toarray).
    """
    #& `vectorizer.transform` devolve uma matriz ESPARSA (CSR): apenas os
    #& não-zeros são armazenados. Na versão antiga, o código chamava
    #& `.toarray()` e materializava uma matriz densa gigante
    #& (4000 x vocabulário), gastando memória e forçando cada produto
    #& escalar a percorrer milhares de zeros inúteis. Mantendo a esparsa,
    #& cada operação toca só os ~50 termos realmente presentes na resenha.
    return vectorizer.transform(textos_preprocessados)