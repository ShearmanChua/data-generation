import os
import sys
import json
import random

from typing import List, Union
import gradio as gr
import pandas as pd

from distilabel.steps.tasks import (
    ChatGeneration,
    Magpie,
    GenerateSentencePair,
    TextGeneration,
)

from distilabel.models import ClientvLLM, OpenAILLM

from prompts.prompts import PROMPT_CREATION_SYS_PROMPT, SYSTEM_PROMPT_W_DOCUMENT_CONTEXT

MAX_NUM_TOKENS = int(os.getenv("MAX_NUM_TOKENS", 2048))
NUM_EVAL_SAMPLES = int(os.getenv("NUM_EVAL_SAMPLES", 10))
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", 5))

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
                         use_magpie: bool = True,
                         progress=gr.Progress()):
        progress(0.1, desc="Initializing...")
        kl_divergence = {}
        progress(0.2, desc="Generating system prompt...")
        system_prompt = self.generate_system_prompt(dataset_description)

        for model, config in self.models:
            if config["client_type"] == "vllm" and use_magpie:
                system_prompt = self.generate_system_prompt(dataset_description, config)
                magpie_num_examples = int(NUM_EVAL_SAMPLES/2)
                text_gen_num_examples = NUM_EVAL_SAMPLES - magpie_num_examples

            elif config["client_type"] == "vllm" or config["client_type"] == "openai":
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
    
    def generate_magpie_data(self,
                             model_config: dict,
                             is_sample: bool,
                             num_samples: int,
                             document_context: List[str],
                             system_prompt: str = "",
                             temperature:float = 0.7, 
                             progress=gr.Progress()):
        
        assert "magpie_pre_query_template" in model_config, "magpie_pre_query_template required in model_config"
        
        if model_config["magpie_pre_query_template"] == "llama3":
            _STOP_SEQUENCES = [
                "<|eot_id|>",
                "<|start_header_id|>",
                "assistant",
                " \n\n",
            ]
        elif model_config["magpie_pre_query_template"] == "qwen2":
            _STOP_SEQUENCES = ["<|im_end|>", "<|im_start|>", "assistant", "\n\n"]
        else:
            _STOP_SEQUENCES = [
                "<|eot_id|>",
                "<|start_header_id|>",
                "assistant",
                " \n\n",
            ]

        generation_kwargs = {
            "temperature": temperature,
            "do_sample": True,
            "max_new_tokens": 256 if is_sample else int(MAX_NUM_TOKENS * 0.25),
            "stop_sequences": _STOP_SEQUENCES,
        }

        total_steps: int = num_samples * 2

        if document_context:
            sampled_context = random.sample(document_context, min(len(document_context), num_samples))
            sampled_system_prompts = [SYSTEM_PROMPT_W_DOCUMENT_CONTEXT.format(
                system_prompt=system_prompt,
                document_context=context) for context in sampled_context]
        else:
            sampled_system_prompts = [system_prompt for _ in range(num_samples)]

        magpie = Magpie(
            llm = self._get_llm(
                generation_kwargs=generation_kwargs,
                magpie_pre_query_template=model_config["magpie_pre_query_template"],
                use_magpie_template=True,
            ),
            n_turns=1,
            output_mappings={"instruction": "prompt", "response": "completion"},
            only_instruction=True
        )

        magpie.load()
        
        # create instructions
        n_processed = 0
        magpie_results = []
        while n_processed < num_samples:
            progress(
                0.5 * n_processed / num_samples,
                total=total_steps,
                desc="(1/2) Generating instructions",
            )
            remaining_rows = num_samples - n_processed
            batch_size = min(batch_size, remaining_rows)
            inputs = [{"system_prompt": prompt} for prompt in sampled_system_prompts]
            batch = list(magpie.process(inputs=inputs))
            magpie_results.extend(batch[0])
            n_processed += batch_size
            
        progress(0.5, desc="(1/2) Generating instructions")
        
        response_results = self.generate_instruction_responses(
            model_config=model_config,
            is_sample=is_sample,
            num_samples=num_samples,
            document_context=document_context,
            system_prompt=system_prompt
        )


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
    
    def generate_instruction_responses(self,
                             model_config: dict,
                             is_sample: bool,
                             num_samples: int,
                             document_context: List[str],
                             instuctions: Union[List[dict], List[str]],
                             system_prompt: str = "",
                             temperature:float = 0.7, 
                             progress=gr.Progress()):
        



    
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
