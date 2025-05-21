import numpy as np
import io
import copy
import json
import torch
from transformers import AutoTokenizer
import random
import pickle
from tqdm import tqdm
from streaming import MDSWriter
import argparse

random.seed(42)

columns = {
    'domain': 'str',
    'indices': 'ndarray:uint32',
    'input_ids': 'ndarray:uint32',
    'mask':  'ndarray:uint8',
    'length': 'int',
}

parser = argparse.ArgumentParser()

group = parser.add_argument_group(title='input data')

group.add_argument('--target_length', type=int, default=65536)
group.add_argument('--total_samples', type=int, default=16000)
group.add_argument('--input_file', type=str, default='longmagpie/fineweb-edu-demo-multidoc10_demo.jsonl')
group.add_argument('--output_dir', type=str,default='longmagpie/fineweb-edu-demo-multidoc10_demo_concat_p_mix')
group.add_argument('--short_dataset', type=str, default='ultrachat') # Replace it with the ultrachat path you downloaded
group.add_argument('--short_data_num', type=int, default=1, help='Number of short data to prepend')
group.add_argument('--choose_rate', type=float, default=0.4)

args = parser.parse_args()

class SelfSFT(torch.utils.data.Dataset):
    """A class for processing a LLama text dataset"""
    def __init__(self,  args):

        self.tokenizer = AutoTokenizer.from_pretrained('Meta-Llama-3-8B-Instruct')

        per_node_samples = args.total_samples
        
        long_data = open(f'{args.input_file}').readlines()
        short_data = open(f'{args.short_dataset}').readlines()

        random.shuffle(long_data)
        random.shuffle(short_data)

        target_length = args.target_length
        short_data_num = args.short_data_num

        doc_ids = []
        doc_mask_ids = []
        ls_indices = []
        last_index = 0 

        long_ptr = 0
        short_ptr = 0

        with MDSWriter(out=args.output_dir, columns=columns, compression=None) as out_stream:
            for _ in tqdm(range(per_node_samples)):
                doc_ids = []
                doc_mask_ids = []
                ls_indices = []
                last_index = 0
                
                
                # First, get data_num samples from short_data
                short_count = 0
                while short_count < short_data_num and short_ptr < len(short_data):
                    short_line = short_data[short_ptr]
                    short_ptr = short_ptr + 1
                    if short_ptr > len(short_data) - 1:
                        random.shuffle(short_data)
                        short_ptr = 0                    
                    try:
                        short_data_item = json.loads(short_line)
                        
                        # Add beginning marker
                        if len(doc_ids) == 0:
                            begin_tokens = self.tokenizer('<|begin_of_text|>', add_special_tokens=False)['input_ids']
                            doc_ids.extend(begin_tokens)
                            doc_mask_ids.extend([0] * len(begin_tokens))
                        
                        # Process dialogue data
                        for idx, item in enumerate(short_data_item):
                            if idx % 2 == 0 and item['role'] == 'user':
                                tmp = '<|start_header_id|>user<|end_header_id|>\n\n' + str(item['content'].replace('User: ', '')) + '<|eot_id|>' + '<|start_header_id|>assistant<|end_header_id|>\n\n'
                                human_tokens = self.tokenizer(tmp, add_special_tokens=False)['input_ids']
                                doc_ids.extend(human_tokens)
                                doc_mask_ids.extend([0] * len(human_tokens))
                            elif idx % 2 == 1 and item['role'] == 'assistant':
                                tmp = str(item['content']) + '<|eot_id|>'
                                assistant_tokens = self.tokenizer(tmp, add_special_tokens=False)['input_ids']
                                doc_ids.extend(assistant_tokens)
                                doc_mask_ids.extend([1] * len(assistant_tokens))
                            else:
                                raise ValueError('Incorrect dialogue format')
                        
                        short_count += 1
                    except json.JSONDecodeError:
                        print(f"Warning: Unable to parse JSON line in short dataset: {short_line[:50]}...")
                        continue
                    except Exception as e:
                        print(f"Error processing short data: {str(e)}")
                        continue
                
                # Randomly select data from long_data or short_data until reaching the target length
                while len(doc_ids) < target_length:
                    # Randomly decide which dataset to use
                    use_long =  random.random() < args.choose_rate
                    
                    if use_long:

                        if long_ptr > len(long_data) - 1:
                            random.shuffle(long_data)
                            long_ptr = 0
                        data = json.loads(long_data[long_ptr])
                        long_ptr += 1

                        
                    else:
                        
                        if short_ptr > len(short_data) - 1:
                            random.shuffle(short_data)
                            short_ptr = 0
                        data = json.loads(short_data[short_ptr])
                        short_ptr += 1

                    
                    # If document is empty, add beginning marker
                    if len(doc_ids) == 0:
                        begin_tokens = self.tokenizer('<|begin_of_text|>', add_special_tokens=False)['input_ids']
                        doc_ids.extend(begin_tokens)
                        doc_mask_ids.extend([0] * len(begin_tokens))
                    
                    # Process dialogue data
                    for idx, item in enumerate(data):
                        if idx % 2 == 0 and item['role'] == 'user':
                            tmp = '<|start_header_id|>user<|end_header_id|>\n\n' + str(item['content'].replace('User: ', '')) + '<|eot_id|>' + '<|start_header_id|>assistant<|end_header_id|>\n\n'
                            human_tokens = self.tokenizer(tmp, add_special_tokens=False)['input_ids']
                            doc_ids.extend(human_tokens)
                            doc_mask_ids.extend([0] * len(human_tokens))
                        elif idx % 2 == 1 and item['role'] == 'assistant':
                            tmp = str(item['content']) + '<|eot_id|>'
                            assistant_tokens = self.tokenizer(tmp, add_special_tokens=False)['input_ids']
                            doc_ids.extend(assistant_tokens)
                            doc_mask_ids.extend([1] * len(assistant_tokens))
                        else:
                            raise ValueError('Incorrect dialogue format')
                
                # Ensure consistent lengths
                assert len(doc_ids) == len(doc_mask_ids)
                
                # If target length is reached or exceeded, save the sample
                if len(doc_ids) >= target_length:
                    ls_indices.append([last_index, target_length])
                    
                    sample = {
                        'domain': "",
                        'indices': np.array(ls_indices, dtype=np.uint32),
                        'input_ids': np.array(doc_ids[:target_length], dtype=np.uint32),
                        'mask': np.array(doc_mask_ids[:target_length], dtype=np.uint8),
                        'length': target_length,
                    }
                    
                    out_stream.write(sample)

SelfSFT(args)

