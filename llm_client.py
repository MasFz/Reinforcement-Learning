from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

class LLMClient:
    def __init__(self, code_model_name='Salesforce/codegen-350M-mono', feedback_model_name='microsoft/DialoGPT-medium'):
        # Carregar o tokenizer e o modelo para geração de código
        self.code_tokenizer = AutoTokenizer.from_pretrained(code_model_name)
        self.code_model = AutoModelForCausalLM.from_pretrained(code_model_name)

        # Carregar o tokenizer e o modelo para geração de feedback
        self.feedback_tokenizer = AutoTokenizer.from_pretrained(feedback_model_name)
        self.feedback_model = AutoModelForCausalLM.from_pretrained(feedback_model_name)

        # Definir o pad_token se não estiver definido
        if self.code_tokenizer.pad_token is None:
            self.code_tokenizer.pad_token = self.code_tokenizer.eos_token
        if self.feedback_tokenizer.pad_token is None:
            self.feedback_tokenizer.pad_token = self.feedback_tokenizer.eos_token

        # Definir o pad_token_id nos modelos
        self.code_model.config.pad_token_id = self.code_tokenizer.pad_token_id
        self.feedback_model.config.pad_token_id = self.feedback_tokenizer.pad_token_id

    def generate_code(self, prompt):
        try:
            # Preparar o prompt
            inputs = self.code_tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)

            # Gerar código
            output = self.code_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=500,
                num_return_sequences=2,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.7
            )

            # Decodificar o código gerado
            code = self.code_tokenizer.decode(output[0], skip_special_tokens=True)

            # Remover o prompt inicial do código gerado
            code = code[len(prompt):].strip()

            # Imprimir o código gerado
            print("Código gerado:\n", code)
            return code
        except Exception as e:
            print(f"Erro ao gerar o código: {e}")
            return ""

    def generate_feedback(self, prompt):
        try:
            # Preparar o prompt
            inputs = self.feedback_tokenizer(prompt + self.feedback_tokenizer.eos_token, return_tensors='pt', padding=True, truncation=True, max_length=512)

            # Gerar feedback
            output = self.feedback_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=100,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.7
            )

            # Decodificar o feedback gerado
            feedback = self.feedback_tokenizer.decode(output[0], skip_special_tokens=True)

            # Remover o prompt inicial do feedback gerado
            feedback = feedback[len(prompt):].strip()

            # Imprimir o feedback gerado
            print("Feedback gerado:\n", feedback)
            return feedback
        except Exception as e:
            print(f"Erro ao gerar o feedback: {e}")
            return ""
