import os
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from loader import LABELS


def get_model(save_path):
    config = AutoConfig.from_pretrained(
        os.path.join(save_path, "config.json")
    )
    return AutoModelForTokenClassification.from_pretrained(
        os.path.join(save_path, "pytorch_model.bin"), config=config
    )


def save_model_info(model_name, save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    tokenizer.save_pretrained(save_path)

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = len(LABELS)
    config.save_pretrained(save_path)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, config=config
    )
    model.save_pretrained(save_path)
    
