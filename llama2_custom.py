
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

print(f'Using device: {device}')

def prompt_creator(text):
    system = """
        <s>[INST] <<SYS>>
        I'm using some computer vision models that extract some characteristics from a photo, such as objects, landscapes and texts.
        There is 2 predictors, Panoptic and OCR, your objective is to use these characteristics to describe the scene in detail.
        You cant ask nothing, just describe the scene in detail. Talk about the objects, the landscape and the texts. Nothing more.
        <</SYS>>
    """
    return f"<s>[INST] <<SYS>> {system} <</SYS>>\n{text}\n[/INST]"


def llama2(prompt):
    if not torch.cuda.is_available():
        print("can't run LlaMA model")
        return None
    else:
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.use_default_system_prompt = False
        # tokenizer.use_default_system_prompt = False

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map=device,
        )

        sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=400,
        )

        printable_sequences = []

        for seq in sequences:
            printable_sequences.append(seq['generated_text'])
            print(f"{seq['generated_text']}")
        
    
        return ' '.join(printable_sequences)
