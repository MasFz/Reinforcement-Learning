import numpy as np
import pandas as pd

class DataCleaningEnv:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.reset()

    def reset(self):
        # Inicializa o ambiente com um DataFrame sintético
        self.df = pd.DataFrame(np.random.randint(0, 100, size=(5, 4)), columns=list('ABCD'))
        self.df = self.df.mask(np.random.random(self.df.shape) < 0.2)
        self.state = self._get_state()
        # Inicializa o atributo missing_percent_before_feedback
        self.missing_percent_before_feedback = self.df.isnull().mean().mean()
        return self.state

    def _get_state(self):
        # Retorna uma representação do estado atual (por exemplo, percentagem de valores faltantes)
        missing_percent = self.df.isnull().mean().mean()
        return np.array([missing_percent])

    def step_coder(self, code):
        print("Executando código:\n", code)
        try:
            # Verifica se o código contém chamadas não permitidas como 'input()'
            if 'input(' in code:
                raise ValueError("O código gerado contém chamadas para input(), que não são permitidas.")

            # Compila o código para verificar a sintaxe
            compiled_code = compile(code, '<string>', 'exec')
            # Cria um namespace seguro para execução
            safe_globals = {'df': self.df.copy(), 'pd': pd, 'np': np, '__builtins__': None}
            safe_locals = {}
            # Executa o código
            exec(compiled_code, safe_globals, safe_locals)
            # Atualiza o DataFrame após a execução do código
            if 'df' in safe_globals:
                self.df = safe_globals['df']
            elif 'df' in safe_locals:
                self.df = safe_locals['df']
            else:
                print("O código não modificou o DataFrame.")
            # Atualiza o estado
            self.state = self._get_state()
            # Atualiza missing_percent_before_feedback
            self.missing_percent_before_feedback = self.df.isnull().mean().mean()
            reward = self._calculate_reward_coder()
            done = self._check_done()
            return self.state, reward, done, {}
        except SyntaxError as e:
            print(f"Erro de sintaxe no código gerado: {e}")
            # Penaliza por código inválido
            reward = -1
            done = True
            return self.state, reward, done, {}
        except Exception as e:
            print(f"Erro ao executar o código gerado: {e}")
            # Penaliza por erro de execução
            reward = -0.5
            done = False
            return self.state, reward, done, {}

    def step_reviewer(self, feedback):
        print("Feedback recebido:", feedback)
        # Processa o feedback do revisor (aqui, não fazemos nada com o feedback)
        reward = self._calculate_reward_reviewer()
        done = self._check_done()
        return self.state, reward, done, {}

    def _calculate_reward_coder(self):
        # Calcula a recompensa para o codificador
        missing_percent = self.df.isnull().mean().mean()
        reward = -missing_percent  # Recompensa negativa proporcional aos valores faltantes
        return reward

    def _calculate_reward_reviewer(self):
        # Calcula a recompensa para o revisor
        missing_percent_after = self.df.isnull().mean().mean()
        improvement = self.missing_percent_before_feedback - missing_percent_after
        reward = improvement * 10  # Escala a melhoria para ajustar a recompensa
        return reward

    def _check_done(self):
        # Verifica se o episódio terminou
        return False
