from transformers import LlamaTokenizer, LlamaForCausalLM, set_seed
import torch
import argparse
import json
import os
from tqdm import tqdm

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
VICUNA_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} ASSISTANT:"

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--prompt",
        type=str,
        default='vicuna',
        help="wiz, alpaca, vicuna",
    )
    parser.add_argument(
        "--model_name_tag",
        type=str,
        default='name',
        help="",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="")
    parser.add_argument("--top_p", type=float, default=1.0, help="")
    parser.add_argument("--do_sample", type=bool, default=True, help="")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # Made the config
    import yaml
    model_configs_path = 'src/alpaca_eval/models_configs'
    model_configs_path_current = os.path.join(model_configs_path,args.model_name_tag)
    os.mkdir(model_configs_path_current)

    with open("inference/model_configs_example.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
    new_dict = {}
    new_dict[args.model_name_tag] = data['vicuna-7b']
    new_dict[args.model_name_tag]['pretty_name'] = args.model_name_tag
    new_dict[args.model_name_tag]['completions_kwargs']['model_name'] = './'+args.model_name_tag

    model_configs_path_current_yaml = os.path.join(model_configs_path_current,'configs.yaml')
    with open(model_configs_path_current_yaml, 'w') as outfile:
        yaml.safe_dump(new_dict, outfile, default_flow_style=False)

    
    # Do inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, cache_dir="../cache/")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, cache_dir="../cache/")

    model.to(device)
    model.eval()

    if args.prompt == 'alpaca':
        prompt_no_input = PROMPT_DICT["prompt_no_input"]
    elif args.prompt == 'wiz':
        prompt_no_input = "{instruction}\n\n### Response:"
    elif args.prompt == 'vicuna':
        prompt_no_input = VICUNA_PROMPT
    
    dataset_path = 'inference/alpaca_eval_data.jsonl'
    prompt_key = 'instruction'
    
    with open(dataset_path) as f:
        results = []
        dataset = list(f)
        for point in tqdm(dataset):
            point = json.loads(point)
            instruction = point[prompt_key]
            prompt = prompt_no_input.format_map({"instruction":instruction})
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            generate_ids = model.generate(
                input_ids, 
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample
                )
            outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            new_point = {}
            new_point["dataset"] = point["dataset"]
            new_point["instruction"] = point["instruction"]
            new_point["output"] = outputs.split("Response:")[1]
            new_point["generator"] = args.model_name_tag

            results.append(new_point)

    output_dir = os.path.join('results',args.model_name_tag)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_name = 'model_outputs.json'
    save_path = os.path.join(output_dir, save_name)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    # Finish inference, can do evaluation now
    os.system("alpaca_eval --model_outputs {} --annotators_config 'alpaca_eval_gpt4' ".format(save_path))


if __name__ == "__main__":
    main()