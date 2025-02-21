import os
import sys
import json

from typing import List
import gradio as gr
import pandas as pd

from distilabel.steps.tasks import (
    ChatGeneration,
    Magpie,
    GenerateSentencePair,
    TextGeneration,
)

from distilabel.models import ClientvLLM, OpenAILLM

from prompts.prompts import PROMPT_CREATION_SYS_PROMPT

MAX_NUM_TOKENS = int(os.getenv("MAX_NUM_TOKENS", 2048))
NUM_EVAL_SAMPLES = int(os.getenv("NUM_EVAL_SAMPLES", 10))

model_configs_file = os.getenv("MODEL_CONFIGS", "../configs/model_configs.json")

class InstructPipeline():
    def __init__(self):
        with open(model_configs_file, "r") as file:
            self.models = json.load(file)
        self.best_model = None
    def generate_samples(self, 
                         dataset_description:str, 
                         example_instructions: pd.DataFrame = None, 
                         document_context: List[str] = [],
                         num_samples: int = 5,
                         progress=gr.Progress()):
        progress(0.1, desc="Initializing...")
        kl_divergence = {}
        progress(0.2, desc="Generating system prompt...")
        system_prompt = self.generate_system_prompt(dataset_description)

        for model, config in self.models:
            if config["client_type"] == "vllm":
                system_prompt = self.generate_system_prompt(dataset_description, config)
                magpie_num_examples = int(NUM_EVAL_SAMPLES/2)
                text_gen_num_examples = NUM_EVAL_SAMPLES - magpie_num_examples

            elif config["client_type"] == "openai":
                system_prompt = self.generate_system_prompt(dataset_description, config)
                text_gen_num_examples = NUM_EVAL_SAMPLES

    def generate_system_prompt(self, 
                               dataset_description: str,
                               model_config: dict):
        generation_kwargs = {
            "temperature": 0.8,
            "max_new_tokens": MAX_NUM_TOKENS,
            "do_sample": True,
        }
        sys_prompt_generator = TextGeneration(
            llm=self._get_llm(generation_kwargs=generation_kwargs, model_config=model_config),
            system_prompt=PROMPT_CREATION_SYS_PROMPT,
            use_system_prompt=True,
        )
        sys_prompt_generator.load()
        generate_description = sys_prompt_generator
        
        result = next(
            generate_description.process(
                [
                    {
                        "instruction": dataset_description,
                    }
                ]
            )
        )[0]["generation"]

        return result
    
    def generate_magpie_data(self):
        pass

    def generate_text_generation_data(self,
                                      model_config: dict,
                                      is_sample: bool,
                                      num_samples: int,
                                      document_context: List[str],
                                      example_instructions: pd.DataFrame = None,
                                      system_prompt: str = "",
                                      temperature:float = 0.7, 
                                      progress=gr.Progress()):
        
        progress(0.1, desc="Initializing...")
        generation_kwargs = {
            "temperature": temperature,
            "max_new_tokens": 256 if is_sample else int(MAX_NUM_TOKENS * 0.5),
        }
        
        
        return text_generation_data
    
    def _get_llm(self,
                 structured_output: dict = None,
                 use_magpie_template: str = False,
                 **kwargs
                ):
        
        model_configs = kwargs["model_config"]
        del kwargs["model_config"]

        if model_configs["client_type"] == "openai":
            llm = OpenAILLM(
                model=model_configs["model"],
                base_url=model_configs["base_url"],
                api_key=model_configs["api_key"],
                structured_output=structured_output,
                **kwargs,
            )
            return llm

        elif model_configs["client_type"] == "vllm":
            assert "tokenizer" in model_configs, "When using vLLM models ensure tokenizer is provide e.g. 'meta-llama/Llama-3.3-70B-Instruct'"
            if "generation_kwargs" in kwargs:
                if "do_sample" in kwargs["generation_kwargs"]:
                    del kwargs["generation_kwargs"]["do_sample"]
            llm = ClientvLLM(
                base_url=model_configs["base_url"],
                model=model_configs["model"],
                tokenizer=model_configs["tokenizer"],
                api_key=model_configs["api_key"],
                use_magpie_template=use_magpie_template,
                structured_output=structured_output,
                **kwargs,
            )

            return llm
        else:
            raise AssertionError("client type of model not currently supported.")
