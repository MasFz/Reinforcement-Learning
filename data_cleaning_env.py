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

    def step_coder(self, code, action_index):
        fallback_actions = {
            0: "df.dropna(inplace=True)",  # Remove rows with missing values
            1: "df.fillna(df.mean(), inplace=True)",  # Fill missing values with column mean
            2: "df.fillna(0, inplace=True)",  # Fill missing values with zero
            3: "df.dropna(axis=1, thresh=int(0.8 * df.shape[0]), inplace=True)"  # Drop columns with many missing values
        }

        print("Executando código:\n", code)
        try:
            # Attempt to execute the generated code
            if 'input(' in code:
                raise ValueError("O código gerado contém chamadas para input(), que não são permitidas.")

            compiled_code = compile(code, '<string>', 'exec')
            safe_globals = {'df': self.df.copy(), 'pd': pd, 'np': np, '__builtins__': None}
            safe_locals = {}
            exec(compiled_code, safe_globals, safe_locals)

            # Update DataFrame if modified
            if 'df' in safe_globals:
                self.df = safe_globals['df']
            elif 'df' in safe_locals:
                self.df = safe_locals['df']
            else:
                print("O código não modificou o DataFrame.")

            reward = self._calculate_reward_coder()
            done = self._check_done()
            return self.state, reward, done, {}

        except (SyntaxError, ValueError) as e:
            print(f"Erro no código gerado. Executando fallback para ação {action_index}")
            fallback_code = fallback_actions.get(action_index, "")
            compiled_code = compile(fallback_code, '<string>', 'exec')
            exec(compiled_code, {'df': self.df, 'pd': pd, 'np': np})
            
            # After fallback, calculate reward
            reward = self._calculate_reward_coder()
            done = self._check_done()
            return self.state, reward, done, {}

        except Exception as e:
            print(f"Erro ao executar o código gerado ou fallback: {e}")
            reward = -0.75  # General penalty for other errors
            done = False
            return self.state, reward, done, {}



    def step_reviewer(self, feedback):
        print("Feedback recebido:", feedback)
        
        # Measure the missing value percentage after coder's action
        missing_percent_after = self.df.isnull().mean().mean()
        
        # Calculate the reward based on improvement
        if missing_percent_after < self.missing_percent_before_feedback:
            reward = (self.missing_percent_before_feedback - missing_percent_after) * 10  # Positive reward for improvement
        else:
            reward = -0.5  # Small penalty if no improvement

        # Update `missing_percent_before_feedback` for the next round
        self.missing_percent_before_feedback = missing_percent_after
        
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
        print(f"Before feedback: {self.missing_percent_before_feedback}, After feedback: {missing_percent_after}")
   
        reward = improvement * 10  # Escala a melhoria para ajustar a recompensa
        return reward

    def _check_done(self):
        # Verifica se o episódio terminou
        return False
