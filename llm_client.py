class LLMClient:
    def __init__(self, api_key):
        self.api_key = api_key
        # Talvez a gente precise configurar a API qualquer coisa a gente coloca aqui!

    def generate_code(self, prompt):
        # Implementar chamada à API para gerar código de limpeza com base no prompt
        # Exemplo de simulação:
        if "remover linhas" in prompt:
            code = "df = df.dropna()"
        elif "remover colunas" in prompt:
            code = "threshold = int(0.67 * len(df)); df = df.dropna(axis=1, thresh=threshold)"
        elif "interpolar" in prompt:
            code = "df = df.interpolate()"
        elif "preencher com média" in prompt:
            code = "df = df.fillna(df.mean())"
        else:
            code = ""
        return code

    def generate_feedback(self, prompt):
        # Implementar chamada à API para gerar feedback com base no DataFrame limpo
        # Exemplo de simulação:
        if "muito missing" in prompt:
            feedback = "Considere remover colunas com muitos valores faltantes."
        elif "boa limpeza" in prompt:
            feedback = "A limpeza dos dados está adequada."
        else:
            feedback = "Verifique se todos os valores faltantes foram tratados."
        return feedback
