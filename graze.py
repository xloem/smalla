"""LLaMA grazing.
A model is exposed to extensive environment and the behavior recorded for further engagement.

Usage:
  graze.py <model_name_or_path> [--dtype=<dtype>] [--batch-size=<size>] <dataset> [--column=<name>] [--tokenizer=<name_or_path>] <output>

Options:
  --dtype=<dtype>  torch_dtype to load the model in [default: float16].
  --batch-size=<size>  [default: 1]
  --tokenizer=<name_or_path>  [default: model_name_or_path].
  --column=<name>  Which column of the dataset to graze on [default: "text" or longest].
"""

_config = None
_tokenizer = None
def tokenize_examples(examples, column, model_name_or_path, tokenizer_name_or_path):
    global _config, _tokenizer
    import transformers
    if _config is None:
        _config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    if _tokenizer is None:
        _tokenizer = transformers.LlamaTokenizer.from_pretrained(tokenizer_name_or_path)
        if _tokenizer.pad_token_id is None:
            _tokenizer.pad_token_id = 0
    return _tokenizer(
        examples[column],
        return_tensors='np',
        truncation=True,
        padding='max_length',
        max_length=_config.max_position_embeddings,
    )

_model = None
_model_device = None
_position_ids = None
def forward_examples(examples, model_name_or_path, dtype):
    global _config, _model, _model_device
    import transformers
    if _config is None:
        _config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    if _model is None:
        import torch
        dtype = getattr(torch, dtype)
        #max_memory = accelerate.utils.get_max_memory()
        ## before this, it ran out of memory on layer idx 2
        #max_memory[0] = 1024*1024*1024#-= 4 * batch_size * config.num_attention_heads * config.max_position_embeddings * config.max_position_embeddings * 2
        _model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype, device_map='auto')#, max_memory=max_memory)
        _model_device = next(_model.parameters()).device
        #assert model_device.index == 0 # this was the device memory was freed on
        _position_ids = torch.arange(_config.max_position_embeddings, device=_model_device, dtype=int)
        _model.eval()
    #input_ids = torch.stack([torch.from_numpy(x) for x in examples['input_ids']]).to(model_device)
    #attention_mask = torch.stack([torch.from_numpy(x) for x in examples['attention_mask']]).to(model_device)
    input_ids = torch.tensor(examples['input_ids'])#, device=_model_device)
    attention_mask = torch.tensor(examples['attention_mask'], device=_model_device)
    #labels = input_ids.clone()
    #labels[attention_mask == 0] = -100
    outputs = _model(
        input_ids = input_ids,
        attention_mask = attention_mask,
        position_ids = _position_ids[:input_ids.shape[-1]],
        output_attentions = True,
        output_hidden_states = True,
        #labels = labels,
        use_cache = False,
        return_dict = True
    )
    for key, val in outputs.items():
        if type(val) is tuple:
            outputs[key] = [#torch.stack(val, dim=1).numpy()
                [
                    layer[batch_idx].numpy()
                    for layer in val
                ]
                for batch_idx in range(val[0].shape[0])
            ]
        else:
            outputs[key] = val.numpy()
    return outputs

def graze(
    model_name_or_path,
    dataset,
    output,
    dtype='float16', batch_size=5,
    column='"text" or longest', tokenizer_name_or_path='model_name_or_path'
):
    import tqdm
    with tqdm.tqdm(total=4, desc='Importing python modules') as pbar:
        import torch, torch.nn as nn, torch.multiprocessing as mp
        pbar.update()
        import transformers
        pbar.update()
        import datasets, accelerate
        pbar.update()
        import sentencepiece, safetensors
        pbar.update()

    getattr(torch, dtype)

    if column == '"text" or longest':
        task = 'language-modeling'
        column = 'text'
    else:
        task = None
    try:
        dataset = datasets.load_dataset(*dataset.split(':',1), save_infos=True, task=task, num_proc=2)
    except ValueError as error:
        assert column == 'text'
        dataset = datasets.load_dataset(*dataset.split(':',1), save_infos=True, num_proc=2)
        _, column = max([(len(val), key) for key, val in next(iter(next(iter(dataset.values())))).items()])

    if tokenizer_name_or_path == 'model_name_or_path':
        tokenizer_name_or_path = model_name_or_path
    dataset = dataset.map(
        tokenize_examples,
        fn_kwargs = {
            'column': column,
            'model_name_or_path': model_name_or_path,
            'tokenizer_name_or_path': tokenizer_name_or_path,
        },
        desc='Tokenizing into feed',
        batched = True,
        num_proc = mp.cpu_count(),
        writer_batch_size = mp.cpu_count() * 1000,
    )

    with torch.no_grad():
        dataset = dataset.map(
            forward_examples,
            fn_kwargs = {
                'model_name_or_path': model_name_or_path,
                'dtype': dtype,
            },
            desc = 'Grazing on tokens',
            batched = True,
            batch_size = batch_size,
        )

        dataset.push_to_hub(output)
    

if __name__ == '__main__':
    import docopt
    params = docopt.docopt(__doc__, version='LLaMA grazing')
    
    graze(
        model_name_or_path = params['<model_name_or_path>'],
        dtype = params['--dtype'],
        batch_size = int(params['--batch-size']),
        dataset = params['<dataset>'],
        column = params['--column'],
        tokenizer_name_or_path = params['--tokenizer'],
        output = params['<output>'],
    )
