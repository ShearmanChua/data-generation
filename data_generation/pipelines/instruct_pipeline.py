import os
import sys
import json

from typing import List
import gradio as gr
import pandas as pd

model_configs_file = os.getenv("MODEL_CONFIGS", "../configs/model_configs.json")

class InstructPipeline():
    def __init__(self):
        with open(model_configs_file, "r") as file:
            self.models = json.load(file)
        self.best_model = None
    def generate_samples(self, dataset_description:str, 
                         example_instructions: pd.DataFrame, 
                         document_context: List[str]):
        pass
    
    def generate_magpie_data(self):
        pass

    def generate_text_generation_data(self, model_config: dict,
                                    document_context: List[str],
                                    system_prompt: str = ""):
        