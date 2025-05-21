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
group.add_argument('--input_file', type=str, default='longmagpie/fineweb-edu-demo-multidoc10_demo.jsonl')
group.add_argument('--output_dir', type=str,default='longmagpie/fineweb-edu-demo-multidoc10_demo_concat')

args = parser.parse_args()

class SelfSFT(torch.utils.data.Dataset):
    
    """A class for processing a LLama text dataset"""
    def __init__(self,  args):

        self.tokenizer = AutoTokenizer.from_pretrained('Meta-Llama-3-8B-Instruct')

        all_data = open(f'{args.input_file}').readlines()

        print(len(all_data))

        target_length = args.target_length

        dic_num = {}

        doc_ids = []
        doc_mask_ids = []
        ls_indices = []

        last_index = 0 

        with MDSWriter(out=args.output_dir, columns=columns, compression=None) as out_stream:

            for line in tqdm(all_data):
                
                data= json.loads(line)

                item_index = 0

                doc_ids = doc_ids + self.tokenizer('<|begin_of_text|>',add_special_tokens=False)['input_ids']

                doc_mask_ids =  doc_mask_ids +  [0] * len(self.tokenizer('<|begin_of_text|>',add_special_tokens=False)['input_ids'])

                for idx, item in enumerate(data):
                    
                    if idx%2==0 and item['role']=='user':
                        tmp='<|start_header_id|>user<|end_header_id|>\n\n' + str(item['content'].replace('User: ','')) + '<|eot_id|>' + '<|start_header_id|>assistant<|end_header_id|>\n\n'
                        human_tokens = self.tokenizer(tmp,add_special_tokens=False)['input_ids']
                        doc_ids = doc_ids + human_tokens
                        doc_mask_ids = doc_mask_ids + [0] * len(human_tokens)

                    elif idx%2==1 and item['role']=='assistant' :
                        tmp= str(item['content']) + '<|eot_id|>'
                        assistant_tokens = self.tokenizer(tmp,add_special_tokens=False)['input_ids']
                        doc_ids = doc_ids + assistant_tokens
                        doc_mask_ids = doc_mask_ids + [1] * len(assistant_tokens)
                        
                    else :
                        raise ValueError('error')

                assert len(doc_ids) == len(doc_mask_ids)

                if len(doc_ids) >= target_length:

                    ls_indices.append([last_index,target_length])
                    
                    sample = {
                        'domain':"",
                        'indices': np.array(ls_indices, dtype=np.uint32),
                        'input_ids': np.array(doc_ids[:target_length], dtype=np.uint32),
                        'mask':  np.array(doc_mask_ids[:target_length], dtype=np.uint8),
                        'length': target_length,
                    }

                    out_stream.write(sample)

                    last_index =  0
                    ls_indices=[]
                    doc_ids = []
                    doc_mask_ids = []

                else :
                    ls_indices.append([last_index, len(doc_ids)])
                    last_index = len(doc_ids)


SelfSFT(args)
