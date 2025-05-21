import os, json
import argparse
from tqdm import tqdm
import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def main():

    os.makedirs(args.save_dir, exist_ok=True)
    
    print(args)

    output_path = os.path.join(args.save_dir, "fineweb-edu-demo_output.jsonl")

    out_file = open(output_path, 'w+')

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm_model = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len)

    input_file = args.input_file

    lines = open(input_file).readlines()

    max_len = int(args.max_model_len // 2)

    sampling_params = SamplingParams(temperature=0.8, max_tokens=int(max_len//2))

    for line in tqdm(lines):

        data = json.loads(line)
        
        context = data['text']

        input_ids = tokenizer.encode(context)

        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            context = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        if 'qwen' in args.model.lower():
            inputs_pre = "<|im_start|>system\n" + context + "<|im_end|>\n" + "<|im_start|>user\n"
        elif 'llama' in args.model.lower():
            inputs_pre = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + context + "\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        
        outputs = llm_model.generate([inputs_pre], sampling_params)

        query = outputs[0].outputs[0].text

        match = re.match(r'^(.*)\?[^?]*$', query)

        if match:
            query = match.group(1) + '?'
        else:
            continue
        
        if len(query) > args.query_len:
            continue

        if 'qwen' in args.model.lower():
            inputs = inputs_pre + query + "<|im_end|>\n" + "<|im_start|>assistant\n"
        elif 'llama' in args.model.lower():
            inputs = inputs_pre + query + "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        answer = llm_model.generate([inputs], sampling_params)

        answer = answer[0].outputs[0].text

        data_new = {'context': context, 'query': query, 'answer': answer}

        out_file.write(json.dumps(data_new) + '\n')

    out_file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", "-s", type=str, default="longmagpie")
    parser.add_argument("--model", "-m", type=str, default="Qwen2.5-72B-Instruct")
    parser.add_argument("--input_file", "-i", type=str, default="longmagpie/fineweb-edu-demo.jsonl")
    parser.add_argument("--tensor_parallel_size", "-tps", type=int, default=4)
    parser.add_argument("--query_len", "-ql", type=int, default=1500)
    parser.add_argument("--max_model_len", "-mml", type=int, default=32768)

    args = parser.parse_args()
    
    main()