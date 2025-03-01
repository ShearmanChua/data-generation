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
from distilabel.distiset import Distiset

from datasets import Dataset

from prompts.prompts import PROMPT_CREATION_SYS_PROMPT, SYSTEM_PROMPT_W_DOCUMENT_CONTEXT

MAX_NUM_TOKENS = int(os.getenv("MAX_NUM_TOKENS", 2048))
NUM_EVAL_SAMPLES = int(os.getenv("NUM_EVAL_SAMPLES", 10))
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", 5))
DEFAULT_SAMPLE_INSTRUCTIONS_NUM = int(os.getenv("DEFAULT_SAMPLE_INSTRUCTIONS_NUM", 5))

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

        for model, config in self.models:
            if config["client_type"] == "vllm" and use_magpie:
                system_prompt = self.generate_system_prompt(dataset_description, config)
                magpie_num_examples = int(NUM_EVAL_SAMPLES/2) if len(document_context) > 0 else NUM_EVAL_SAMPLES
                text_gen_num_examples = NUM_EVAL_SAMPLES - magpie_num_examples

                magpie_examples = self.generate_magpie_data(
                    model_config=config,
                    is_sample=True,
                    num_samples=magpie_num_examples,
                    document_context=document_context,
                    system_prompt=system_prompt,
                )

            elif config["client_type"] == "vllm" or config["client_type"] == "openai":
                system_prompt = self.generate_system_prompt(dataset_description, config)
                assert example_instructions is not None or len(document_context) > 0, "example_instructions or document_context required"
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
            if len(sampled_context) < num_samples:
                sampled_system_prompts += [system_prompt for _ in range(num_samples - len(sampled_context))]
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

        batch_size = DEFAULT_BATCH_SIZE
        
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
            batch = sampled_system_prompts[n_processed : n_processed + batch_size]
            inputs = [{"system_prompt": prompt} for prompt in batch]
            batch = list(magpie.process(inputs=inputs))
            magpie_results.extend(batch[0])
            n_processed += batch_size
            
        progress(0.5, desc="(1/2) Generating instructions")

        for instruction, system_prompt in zip(magpie_results, sampled_system_prompts):
            instruction["system_prompt"] = system_prompt
        
        response_results = self.generate_instruction_responses(
            model_config=model_config,
            is_sample=is_sample,
            num_samples=num_samples,
            document_context=document_context,
            instuctions=magpie_results,
            progress=progress
        )

        progress(
            1,
            total=total_steps,
            desc="(2/2) Creating dataset",
        )

        distiset_results = []
        for result in response_results:
            record = {}
            for relevant_keys in [
                "messages",
                "prompt",
                "completion",
                "model_name",
                "system_prompt",
            ]:
                if relevant_keys in result:
                    record[relevant_keys] = result[relevant_keys]
            distiset_results.append(record)

        distiset = Distiset(
            {
                "default": Dataset.from_list(distiset_results),
            }
        )

        distiset = distiset["default"]
        outputs = distiset.to_pandas()[["prompt", "completion", "system_prompt"]]
        dataframe = pd.DataFrame(outputs)
        progress(1.0, desc="Dataset generation completed")
        return dataframe

    def generate_text_generation_data(self,
                                      model_config: dict,
                                      is_sample: bool,
                                      num_samples: int,
                                      document_context: List[str],
                                      example_instructions: pd.DataFrame = None,
                                      system_prompt: str = "",
                                      temperature:float = 0.7, 
                                      progress=gr.Progress()):
        ''
        progress(0.1, desc="Initializing...")
        generation_kwargs = {
            "temperature": temperature,
            "max_new_tokens": 256 if is_sample else MAX_NUM_TOKENS,
        }

        if example_instructions is not None:
            # get random num_samples example instructions from "prompt" column and "completion" column
            sampled_instructions = example_instructions.sample(n=DEFAULT_SAMPLE_INSTRUCTIONS_NUM, random_state=42)[["prompt", "completion"]]

            sampled_prompts = sampled_instructions["prompt"].tolist()
            
            instructuing_generator = TextGeneration(
                llm=self._get_llm(is_completion=True, 
                                generation_kwargs=generation_kwargs),
                output_mappings={"generation": "completion"},
            )
            instructuing_generator.load()

            if document_context:
                sampled_context = random.sample(document_context, min(len(document_context), num_samples))
                sampled_system_prompts = [SYSTEM_PROMPT_W_DOCUMENT_CONTEXT.format(
                    system_prompt=system_prompt,
                    document_context=context) for context in sampled_context]
                if len(sampled_context) < num_samples:
                    sampled_system_prompts += [system_prompt for _ in range(num_samples - len(sampled_context))]
            else:
                sampled_system_prompts = [system_prompt for _ in range(num_samples)]

            text_generation_data = []
            for system_prompt in sampled_system_prompts:
                text_generation_data.append({"examples": sampled_prompts, "system_prompt": system_prompt})
        
        else:
            assert len(document_context) > 0, "document_context required"
            sampled_context = random.sample(document_context, min(len(document_context), num_samples))
            if len(sampled_context) < num_samples:
                sampled_context += [random.choice(document_context) for _ in range(num_samples - len(sampled_context))]
            
            instructuing_generator = GenerateSentencePair(
                llm=self._get_llm(generation_kwargs=generation_kwargs),
                triplet=False,
                action="query",
                hard_negative=True,
            )
            instructuing_generator.load()

            text_generation_data = [{"anchor": context} for context in sampled_context]

        total_steps: int = num_samples * 2

        batch_size = DEFAULT_BATCH_SIZE

        # create instructions
        n_processed = 0
        instructions_results = []
        while n_processed < num_samples:
            progress(
                0.5 * n_processed / num_samples,
                total=total_steps,
                desc="(1/2) Generating instructions",
            )
            remaining_rows = num_samples - n_processed
            batch_size = min(batch_size, remaining_rows)
            batch = text_generation_data[n_processed : n_processed + batch_size]
            batch = list(instructuing_generator.process(inputs=batch))
            instructions_results.extend(batch[0])
            n_processed += batch_size
            
        progress(0.5, desc="(1/2) Generating instructions")

        #TODO: generate responses
        
        return text_generation_data
    
    def generate_instruction_responses(self,
                             model_config: dict,
                             is_sample: bool,
                             num_samples: int,
                             document_context: List[str],
                             instuctions: List[dict],
                             system_prompt: str = "",
                             temperature:float = 0.7, 
                             progress=gr.Progress()):
        
        total_steps: int = num_samples * 2
        batch_size = DEFAULT_BATCH_SIZE

        generation_kwargs = {
            "temperature": temperature,
            "max_new_tokens": 256 if is_sample else int(MAX_NUM_TOKENS * 0.5),
        }
        response_generator = TextGeneration(
            llm=self._get_llm(is_completion=True, generation_kwargs=generation_kwargs),
            output_mappings={"generation": "completion"},
            input_mappings={"instruction": "prompt"},
        )
        response_generator.load()
        # generate responses
        n_processed = 0
        response_results = []
        while n_processed < num_samples:
            progress(
                0.5 + 0.5 * n_processed / num_samples,
                total=total_steps,
                desc="(2/2) Generating responses",
            )
            batch = instuctions[n_processed : n_processed + batch_size]
            responses = list(response_generator.process(inputs=batch))
            response_results.extend(responses[0])
            n_processed += batch_size

        for result in response_results:
            result["prompt"] = result["instruction"]
            result["completion"] = result["generation"]
        
        return response_results
    
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
