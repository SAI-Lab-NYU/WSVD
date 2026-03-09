import math
import time
import tqdm
import torch
import torch.nn as nn
import utils
import quant_utils
import model_utils
import logging
import os

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)
        
        
@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    quantizers = {}
    sequential = []
    if args.svd_modules in ['qkv', 'attn', 'all']:
        sequential.append(['self_attn.k_proj.ALinear.module', 'self_attn.v_proj.ALinear.module', 'self_attn.q_proj.ALinear.module'])
        sequential.append(['self_attn.k_proj.BLinear.module', 'self_attn.v_proj.BLinear.module', 'self_attn.q_proj.BLinear.module'])
    else:
        sequential.append(['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'])
    if args.svd_modules in ['attn', 'all']:
        sequential.append(['self_attn.o_proj.ALinear.module'])
        sequential.append(['self_attn.o_proj.BLinear.module'])
    else:
        sequential.append(['self_attn.o_proj.module'])
    if args.svd_modules in ['mlp', 'gaup']:
        sequential.append(['mlp.up_proj.ALinear.module', 'mlp.gate_proj.ALinear.module'])
        sequential.append(['mlp.up_proj.BLinear.module', 'mlp.gate_proj.BLinear.module'])
    else:
        sequential.append(['mlp.up_proj.module', 'mlp.gate_proj.module'])
    if args.svd_modules in ['all', 'mlp']:
        sequential.append(['mlp.down_proj.ALinear.module'])
        sequential.append(['mlp.down_proj.BLinear.module'])
    else:
        sequential.append(['mlp.down_proj.module'])
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and 'down_proj' in name:
                    layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, \
                        static_groups=False
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers


@torch.no_grad()
def gptq_fwrdllava(model, dataloader, dev, args, tokenizer=None, image_processor=None):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    dataloader, _ = dataloader
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    model.model.mm_projector = model.model.mm_projector.to(dev)
    model.model.vision_tower = model.model.vision_tower.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    # inps = torch.zeros(
    #     (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    # )
    inps = []
    cache = {'i': 0, 'attention_mask': [], 'position_embeddings':[]}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            # inps[cache['i']] = inp
            inps.append(inp[0])
            cache['i'] += 1
            cache['attention_mask'].append(kwargs['attention_mask'].cpu() if kwargs['attention_mask'] is not None else None)
            cache['position_embeddings'].append(kwargs['position_embeddings'])
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            # model(batch[0].to(dev))
            input_ids, images = message_to_prompt(batch, image_processor, model, tokenizer)
            images, image_sizes = images
            from llava.constants import IMAGE_TOKEN_INDEX
            # num_images = (input_ids == IMAGE_TOKEN_INDEX).sum()
            # print(num_images)
            model.generate(input_ids, images=images,
                            do_sample=False,
                            temperature=0,
                            max_new_tokens=512,
                            use_cache=True,)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    model.model.mm_projector = model.model.mm_projector.cpu()
    model.model.vision_tower = model.model.vision_tower.cpu()
    torch.cuda.empty_cache()

    outs = [None] * args.nsamples 
    attention_mask = cache['attention_mask']
    position_embeddings = cache['position_embeddings'] # should be on the fly generated

    quantizers = {}
    sequential = [
                ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
                ['self_attn.o_proj.module'],
                ['mlp.up_proj.module', 'mlp.gate_proj.module'],
                ['mlp.down_proj.module']
            ]
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and 'down_proj' in name:
                    layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                # attention_mask = inps. 
                # position_embeddings = 
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask[j].to(dev) if attention_mask[j] is not None else None, position_embeddings=position_embeddings[j])[0][0]
                
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask[j].to(dev) if attention_mask[j] is not None else None, position_embeddings=position_embeddings[j])[0][0]# as we have list instead of tensor

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers
       

@torch.no_grad()
def gptq_fwrdllavasvd_qkv(model, dataloader, dev, args, tokenizer=None, image_processor=None):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    dataloader, _ = dataloader
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    model.model.mm_projector = model.model.mm_projector.to(dev)
    model.model.vision_tower = model.model.vision_tower.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    # inps = torch.zeros(
    #     (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    # )
    inps = []
    cache = {'i': 0, 'attention_mask': [], 'position_embeddings':[]}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            # inps[cache['i']] = inp
            inps.append(inp[0])
            cache['i'] += 1
            cache['attention_mask'].append(kwargs['attention_mask'].cpu() if kwargs['attention_mask'] is not None else None)
            cache['position_embeddings'].append(kwargs['position_embeddings'])
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            # model(batch[0].to(dev))
            input_ids, images = message_to_prompt(batch, image_processor, model, tokenizer)
            images, image_sizes = images
            from llava.constants import IMAGE_TOKEN_INDEX
            # num_images = (input_ids == IMAGE_TOKEN_INDEX).sum()
            # print(num_images)
            model.generate(input_ids, images=images,
                            do_sample=False,
                            max_new_tokens=1,
                            use_cache=True,)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    model.model.mm_projector = model.model.mm_projector.cpu()
    model.model.vision_tower = model.model.vision_tower.cpu()
    torch.cuda.empty_cache()

    outs = [None] * args.nsamples 
    attention_mask = cache['attention_mask']
    position_embeddings = cache['position_embeddings'] # should be on the fly generated

    quantizers = {}
    sequential = []
    if args.svd_modules in ['qkv', 'attn', 'all']:
        sequential.append(['self_attn.k_proj.ALinear.module', 'self_attn.v_proj.ALinear.module', 'self_attn.q_proj.ALinear.module'])
        sequential.append(['self_attn.k_proj.BLinear.module', 'self_attn.v_proj.BLinear.module', 'self_attn.q_proj.BLinear.module'])
    else:
        sequential.append(['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'])
    if args.svd_modules in ['attn', 'all']:
        sequential.append(['self_attn.o_proj.ALinear.module'])
        sequential.append(['self_attn.o_proj.BLinear.module'])
    else:
        sequential.append(['self_attn.o_proj.module'])
    if args.svd_modules in ['mlp', 'gaup']:
        sequential.append(['mlp.up_proj.ALinear.module', 'mlp.gate_proj.ALinear.module'])
        sequential.append(['mlp.up_proj.BLinear.module', 'mlp.gate_proj.BLinear.module'])
    else:
        sequential.append(['mlp.up_proj.module', 'mlp.gate_proj.module'])
    if args.svd_modules in ['all', 'mlp']:
        sequential.append(['mlp.down_proj.ALinear.module'])
        sequential.append(['mlp.down_proj.BLinear.module'])
    else:
        sequential.append(['mlp.down_proj.module'])
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and 'down_proj' in name:
                    layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                # attention_mask = inps. 
                # position_embeddings = 
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask[j].to(dev) if attention_mask[j] is not None else None, position_embeddings=position_embeddings[j])[0][0]
                
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask[j].to(dev) if attention_mask[j] is not None else None, position_embeddings=position_embeddings[j])[0][0]# as we have list instead of tensor

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers
       


@torch.no_grad()
def gptq_fwrdvit(model, dataloader, dev, args, tokenizer=None, image_processor=None):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ vit Quantization-----')
    dataloader, _ = dataloader
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.vision_tower.vision_tower.vision_model.encoder.layers

    # model.model.embed_tokens = model.model.embed_tokens.to(dev)
    # model.model.norm = model.model.norm.to(dev)
    # model.model.rotary_emb = model.model.rotary_emb.to(dev)
    # model.model.mm_projector = model.model.mm_projector.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.model.vision_tower.vision_tower.vision_model.parameters())).dtype
    # inps = torch.zeros(
    #     (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    # )
    inps = torch.zeros(
        (args.nsamples, model.model.vision_tower.num_patches+1, model.model.vision_tower.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': [], 'position_embeddings':[]}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inps[cache['i']] = inp
            # inps.append(inp[0])
            cache['i'] += 1
            # cache['attention_mask'].append(kwargs['attention_mask'].cpu())
            # cache['position_embeddings'].append(kwargs['position_embeddings'])
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            # model(batch[0].to(dev))
            input_ids, images = message_to_prompt(batch, image_processor, model, tokenizer)
            from llava.constants import IMAGE_TOKEN_INDEX
            images, image_sizes = images
            # num_images = (input_ids == IMAGE_TOKEN_INDEX).sum()
            # print(num_images)
            model.generate(input_ids, images=images,
                            do_sample=False,
                            temperature=0,
                            max_new_tokens=512,
                            use_cache=True,)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    # model.model.embed_tokens = model.model.embed_tokens.cpu()
    # model.model.norm = model.model.norm.cpu()
    # model.model.rotary_emb = model.model.rotary_emb.cpu()
    # model.model.mm_projector = model.model.mm_projector.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    # attention_mask = cache['attention_mask']
    # position_embeddings = cache['position_embeddings'] # should be on the fly generated

    quantizers = {}
    sequential = [
                ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
                ['self_attn.out_proj.module'],
                ['mlp.fc1.module'],
                ['mlp.fc2.module']
            ]
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and 'down_proj' in name:
                    layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                # attention_mask = inps. 
                # position_embeddings = 
                outs[j] = layer(inps[j].unsqueeze(0), None, None)[0]
                
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, \
                        static_groups=False, #blocksize=32,
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), None, None)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ vit Quantization Done-----\n')
    return quantizers


@torch.no_grad()
def gptq_fwrdmm(model, dataloader, dev, args, tokenizer=None, image_processor=None):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ mm Quantization-----')
    dataloader, _ = dataloader
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model

    # model.model.vision_tower.vision_tower.vision_model.encoder.layers = model.model.vision_tower.vision_tower.vision_model.encoder.layers.to(dev)
    model.model.vision_tower = model.model.vision_tower.to(dev)
    layer = layers.mm_projector.to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.model.vision_tower.num_patches, model.model.vision_tower.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': [], 'position_embeddings':[]}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inps[cache['i']] = inp
            # inps.append(inp[0])
            cache['i'] += 1
            # cache['attention_mask'].append(kwargs['attention_mask'].cpu())
            # cache['position_embeddings'].append(kwargs['position_embeddings'])
            raise ValueError
    layers.mm_projector = Catcher(layers.mm_projector)
    for batch in dataloader:
        try:
            # model(batch[0].to(dev))
            input_ids, images = message_to_prompt(batch, image_processor, model, tokenizer)
            from llava.constants import IMAGE_TOKEN_INDEX
            images, image_sizes = images
            # num_images = (input_ids == IMAGE_TOKEN_INDEX).sum()
            # print(num_images)
            model.generate(input_ids, images=images,
                            do_sample=False,
                            temperature=0,
                            max_new_tokens=512,
                            use_cache=True,)
        except ValueError:
            pass
    # layers[0] = layers[0].module
    layers.mm_projector = layers.mm_projector.module # remove catcher

    layer = layer.cpu()
    # model.model.vision_tower.vision_tower.vision_model.encoder.layers = model.model.vision_tower.vision_tower.vision_model.encoder.layers.cpu()
    model.model.vision_tower = model.model.vision_tower.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros(
        (args.nsamples, model.model.vision_tower.num_patches, model.model.config.hidden_size), dtype=dtype, device=dev
    )
    # attention_mask = cache['attention_mask']
    # position_embeddings = cache['position_embeddings'] # should be on the fly generated

    quantizers = {}
    sequential = [
                ['0.module'],
                ['2.module'],
            ]

    layer = layer.to(dev)
    full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
    for names in sequential:
        subset = {n: full[n] for n in names}

        gptq = {}
        for name in subset:
            print(f'{name}', end='  ', flush=True)
            layer_weight_bits = args.w_bits
            layer_weight_sym = not(args.w_asym)
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = quant_utils.WeightQuantizer()
            gptq[name].quantizer.configure(
                layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            # attention_mask = inps. 
            # position_embeddings = 
            outs[j] = layer(inps[j].unsqueeze(0))[0]
            
        for h in handles:
            h.remove()

        for name in subset:
            layer_w_groupsize = args.w_groupsize
            gptq[name].fasterquant(
                percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
            )
            quantizers['model.model.mm_projector.%s' % (name)] = gptq[name].quantizer
            gptq[name].free()

    # for j in range(args.nsamples):
    #     outs[j] = layer(inps[j].unsqueeze(0))[0]

    layers.mm_projector = layer.cpu()
    del layer
    del gptq 
    torch.cuda.empty_cache()

    inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ mm Quantization Done-----\n')
    return quantizers


@torch.no_grad()
def gptq_fwrdllavasvd_qkvhf(model, dataloader, dev, args, tokenizer=None, image_processor=None):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    dataloader, _ = dataloader
    layers = model.language_model.model.layers

    model.language_model.model.embed_tokens = model.language_model.model.embed_tokens.to(dev)
    model.language_model.model.norm = model.language_model.model.norm.to(dev)
    model.language_model.model.rotary_emb = model.language_model.model.rotary_emb.to(dev)
    model.multi_modal_projector = model.multi_modal_projector.to(dev)
    model.vision_tower = model.vision_tower.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    # inps = torch.zeros(
    #     (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    # )
    inps = []
    cache = {'i': 0, 'attention_mask': [], 'position_embeddings':[]}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            # inps[cache['i']] = inp
            inps.append(inp[0])
            cache['i'] += 1
            cache['attention_mask'].append(kwargs['attention_mask'].cpu() if kwargs['attention_mask'] is not None else None)
            cache['position_embeddings'].append(kwargs['position_embeddings'])
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            # model(batch[0].to(dev))
            input_ids, images = message_to_prompt(batch, image_processor, model, tokenizer)
            inputs = input_ids.to(dev)
            # num_images = (input_ids == IMAGE_TOKEN_INDEX).sum()
            # print(num_images)
            model.generate(**inputs,
                            max_new_tokens=1,
                            use_cache=True,)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model = model.cpu()
    torch.cuda.empty_cache()

    outs = [None] * args.nsamples 
    attention_mask = cache['attention_mask']
    position_embeddings = cache['position_embeddings'] # should be on the fly generated

    quantizers = {}
    sequential = []
    if args.svd_modules in ['qkv', 'attn', 'all']:
        sequential.append(['self_attn.k_proj.ALinear.module', 'self_attn.v_proj.ALinear.module', 'self_attn.q_proj.ALinear.module'])
        sequential.append(['self_attn.k_proj.BLinear.module', 'self_attn.v_proj.BLinear.module', 'self_attn.q_proj.BLinear.module'])
    else:
        sequential.append(['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'])
    if args.svd_modules in ['attn', 'all']:
        sequential.append(['self_attn.o_proj.ALinear.module'])
        sequential.append(['self_attn.o_proj.BLinear.module'])
    else:
        sequential.append(['self_attn.o_proj.module'])
    if args.svd_modules in ['mlp', 'gaup']:
        sequential.append(['mlp.up_proj.ALinear.module', 'mlp.gate_proj.ALinear.module'])
        sequential.append(['mlp.up_proj.BLinear.module', 'mlp.gate_proj.BLinear.module'])
    else:
        sequential.append(['mlp.up_proj.module', 'mlp.gate_proj.module'])
    if args.svd_modules in ['all', 'mlp']:
        sequential.append(['mlp.down_proj.ALinear.module'])
        sequential.append(['mlp.down_proj.BLinear.module'])
    else:
        sequential.append(['mlp.down_proj.module'])
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and 'down_proj' in name:
                    layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                # attention_mask = inps. 
                # position_embeddings = 
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask[j].to(dev) if attention_mask[j] is not None else None, position_embeddings=position_embeddings[j])[0][0]
                
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask[j].to(dev) if attention_mask[j] is not None else None, position_embeddings=position_embeddings[j])[0][0]# as we have list instead of tensor

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    # model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers
       
@torch.no_grad()
def gptq_fwrdvithf(model, dataloader, dev, args, tokenizer=None, image_processor=None):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ vit Quantization-----')
    dataloader, _ = dataloader
    # use_cache = model.config.use_cache
    # model.config.use_cache = False
    layers = model.vision_tower.vision_model.encoder.layers

    # model.model.embed_tokens = model.model.embed_tokens.to(dev)
    # model.model.norm = model.model.norm.to(dev)
    # model.model.rotary_emb = model.model.rotary_emb.to(dev)
    # model.model.mm_projector = model.model.mm_projector.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.vision_tower.vision_model.parameters())).dtype
    # inps = torch.zeros(
    #     (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    # )
    inps = []
    cache = {'i': 0, 'attention_mask': [], 'position_embeddings':[]}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inps.append(inp[0])
            # inps.append(inp[0])
            cache['i'] += 1
            # cache['attention_mask'].append(kwargs['attention_mask'].cpu())
            # cache['position_embeddings'].append(kwargs['position_embeddings'])
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            # model(batch[0].to(dev))
            input_ids, images = message_to_prompt(batch, image_processor, model, tokenizer)
            inputs = input_ids.to(dev)
            # num_images = (input_ids == IMAGE_TOKEN_INDEX).sum()
            # print(num_images)
            model.generate(**inputs,
                            max_new_tokens=1,
                            use_cache=True,)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    # model.model.embed_tokens = model.model.embed_tokens.cpu()
    # model.model.norm = model.model.norm.cpu()
    # model.model.rotary_emb = model.model.rotary_emb.cpu()
    # model.model.mm_projector = model.model.mm_projector.cpu()
    torch.cuda.empty_cache()

    outs = [None] * args.nsamples
    # attention_mask = cache['attention_mask']
    # position_embeddings = cache['position_embeddings'] # should be on the fly generated

    quantizers = {}
    sequential = [
                ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
                ['self_attn.out_proj.module'],
                ['mlp.fc1.module'],
                ['mlp.fc2.module']
            ]
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and 'down_proj' in name:
                    layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                # attention_mask = inps. 
                # position_embeddings = 
                outs[j] = layer(inps[j].unsqueeze(0), None, None)[0][0]
                
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, \
                        static_groups=False, #blocksize=32,
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), None, None)[0][0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    # model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ vit Quantization Done-----\n')
    return quantizers


@torch.no_grad()
def gptq_fwrdmmhf(model, dataloader, dev, args, tokenizer=None, image_processor=None):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ mm Quantization-----')
    dataloader, _ = dataloader
    layers = model

    model.vision_tower = model.vision_tower.to(dev)
    layer = layers.multi_modal_projector.to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = []
    cache = {'i': 0, 'attention_mask': [], 'position_embeddings':[]}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inps.append(inp[0])
            # inps.append(inp[0])
            cache['i'] += 1
            # cache['attention_mask'].append(kwargs['attention_mask'].cpu())
            # cache['position_embeddings'].append(kwargs['position_embeddings'])
            raise ValueError
    layers.multi_modal_projector = Catcher(layers.multi_modal_projector)
    for batch in dataloader:
        try:
            # model(batch[0].to(dev))
            input_ids, images = message_to_prompt(batch, image_processor, model, tokenizer)
            inputs = input_ids.to(dev)
            # num_images = (input_ids == IMAGE_TOKEN_INDEX).sum()
            # print(num_images)
            model.generate(**inputs,
                            max_new_tokens=1,
                            use_cache=True,)
        except ValueError:
            pass
    # layers[0] = layers[0].module
    layers.multi_modal_projector = layers.multi_modal_projector.module # remove catcher

    layer = layer.cpu()
    model.vision_tower = model.vision_tower.cpu()
    torch.cuda.empty_cache()

    outs = [None] * args.nsamples
    # attention_mask = cache['attention_mask']
    # position_embeddings = cache['position_embeddings'] # should be on the fly generated

    quantizers = {}
    sequential = [
                ['linear_1.module'],
                ['linear_2.module'],
            ]

    layer = layer.to(dev)
    full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
    for names in sequential:
        subset = {n: full[n] for n in names}

        gptq = {}
        for name in subset:
            print(f'{name}', end='  ', flush=True)
            layer_weight_bits = args.w_bits
            layer_weight_sym = not(args.w_asym)
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = quant_utils.WeightQuantizer()
            gptq[name].quantizer.configure(
                layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            # attention_mask = inps. 
            # position_embeddings = 
            outs[j] = layer(inps[j].unsqueeze(0))[0][0]
            
        for h in handles:
            h.remove()

        for name in subset:
            layer_w_groupsize = args.w_groupsize
            gptq[name].fasterquant(
                percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
            )
            quantizers['model.model.mm_projector.%s' % (name)] = gptq[name].quantizer
            gptq[name].free()

    # for j in range(args.nsamples):
    #     outs[j] = layer(inps[j].unsqueeze(0))[0]

    layers.multi_modal_projector = layer.cpu()
    del layer
    del gptq 
    torch.cuda.empty_cache()

    inps, outs = outs, inps

    # model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ mm Quantization Done-----\n')
    return quantizers



@torch.no_grad()
def rtn_fwrd(model, dev, args, start_id=0):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    assert args.w_groupsize ==-1, "Groupsize not supported in RTN!"
    layers = model_utils.get_layers(model)
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(start_id, len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear, torch.nn.ParameterList]) # SVDLinear
        
        for name in subset:
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and 'down_proj' in name:
                layer_weight_bits = 8

            if 'ALinear' in name: # nn.ParameterList
                for j, uu in enumerate(subset[name]):
                    quantizer = quant_utils.WeightQuantizer()
                    quantizer.configure(
                        layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
                    )
                    with torch.no_grad():
                        # uu is a Parameter
                        W = uu.data
                        quantizer.find_params(W)
                        uu.data.copy_(quantizer.quantize(W).to(next(iter(layer.parameters())).dtype))
                    quantizers['model.layers.%d.%s.%d' % (i, name, j)] = quantizer.cpu()
            else: # nn.Linear
                quantizer = quant_utils.WeightQuantizer()
                quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
                )
                W = subset[name].weight.data
                quantizer.find_params(W)
                subset[name].weight.data = quantizer.quantize(W).to(
                    next(iter(layer.parameters())).dtype)
                quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer
            
    utils.cleanup_memory(verbos=True)
    return quantizers


@torch.no_grad()
def rtn_fwrdvit(model, dev, args, start_id=0):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    assert args.w_groupsize ==-1, "Groupsize not supported in RTN!"
    if type(model) ==  model_utils.LLAVA_NEXT_HF:
        layers = model.vision_tower.vision_model.encoder.layers
    else:
        layers = model.model.vision_tower.vision_tower.vision_model.encoder.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(start_id, len(layers)), desc="(RtN vit Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(layer,
                                            layers=[torch.nn.Linear])

        for name in subset:
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and 'down_proj' in name:
                layer_weight_bits = 8

            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(
                next(iter(layer.parameters())).dtype)
            quantizers['model.vitlayers.%d.%s' % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer
            
    utils.cleanup_memory(verbos=True)
    return quantizers



@torch.no_grad()
def rtn_fwrdmm(model, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    assert args.w_groupsize ==-1, "Groupsize not supported in RTN!"
    
    torch.cuda.empty_cache()

    quantizers = {}
    if type(model) ==  model_utils.LLAVA_NEXT_HF:
        layer = model.multi_modal_projector
        layer = layer.to(dev)
    else:
        layers = model.model
        layer = layers.mm_projector.to(dev)

    subset = quant_utils.find_qlayers(layer,
                                        layers=[torch.nn.Linear])

    for name in subset:
        layer_weight_bits = args.w_bits
        if 'lm_head' in name:
            layer_weight_bits = 16
            continue
        if args.int8_down_proj and 'down_proj' in name:
            layer_weight_bits = 8

        quantizer = quant_utils.WeightQuantizer()
        quantizer.configure(
            layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
        )
        W = subset[name].weight.data
        dtype = W.dtype
        quantizer.find_params(W)
        subset[name].weight.data = quantizer.quantize(W).to(
            dtype)
        quantizers['model.model.mm_projector.%s' % (name)] = quantizer.cpu()
    if type(model) ==  model_utils.LLAVA_NEXT_HF:
        layer = layer.cpu()
    else:
        layers.mm_projector = layer.cpu()
    torch.cuda.empty_cache()
    del layer
            
    utils.cleanup_memory(verbos=True)
    return quantizers

def insert_ignore_index_after_prompt(input_ids, output_ids, image_token_id=32000, ignore_index=-100):
    """
    In output_ids, insert the corresponding number of ignore_index (-100)
    after the prompt section and before the image token section, to mask loss.

    Args:
        input_ids (torch.Tensor): shape (seq_len,)
        output_ids (torch.Tensor): shape (seq_len,)
        image_token_id (int): Image placeholder token id, default 32000 (HF LLaVANextProcessor)
        ignore_index (int): Ignore marker for CrossEntropyLoss, default -100

    Returns:
        torch.Tensor: Processed output_ids with inserted ignore_index segment
    """
    # Find the position of the first <image> token
    image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)
    if len(image_positions[0]) == 0:
        # If no image token exists, return original output_ids directly
        return output_ids.clone()

    first_image_idx = image_positions[0][0].item()
    num_image_tokens = (input_ids == image_token_id).sum().item()

    # Split prompt part and remaining part
    prompt_output_ids = output_ids[:first_image_idx]
    rest_output_ids = output_ids[first_image_idx:]

    # Build ignore_index segment
    ignore_prefix = torch.full((num_image_tokens,), ignore_index, dtype=output_ids.dtype, device=output_ids.device)

    # Concatenate
    final_output_ids = torch.cat([prompt_output_ids, ignore_prefix, rest_output_ids], dim=0)

    return final_output_ids

def insert_ignore_index_for_smolvlm(input_ids, labels, fake_token_id=49152, image_token_id=49153, ignore_index=-100):
    """
    Set label masks for SmolVLM model with its special image token structure.
    Use torch.cat to add ignore_index padding to the left side of labels.
    
    Args:
        input_ids: Input token sequence
        labels: Label sequence
        fake_token_id: Token ID of <fake_token_around_image>
        image_token_id: Token ID of <image>
        ignore_index: Index value used for masking
    """
    # Find positions of <fake_token_around_image> token
    fake_token_positions = (input_ids == fake_token_id).nonzero(as_tuple=True)[0]
    
    if len(fake_token_positions) >= 2:
        # Find end position of image sequence (second fake_token)
        end_pos = fake_token_positions[1]
        
        # Create a padding tensor of length end_pos+1, all filled with ignore_index
        padding = torch.full((end_pos+1,), ignore_index, device=labels.device, dtype=labels.dtype)
        
        # Concatenate padding and original labels with torch.cat
        new_labels = torch.cat([padding, labels], dim=0)
        
        return new_labels
    
    return labels
    
def message_to_prompt(message, image_processor, model, tokenizer):
    from llava.mm_utils import (
        process_images,
        tokenizer_image_token,
        KeywordsStoppingCriteria,
    )
    from llava.constants import IMAGE_TOKEN_INDEX
    from PIL import Image
    if tokenizer is None:
        from transformers.image_utils import load_image
        from PIL import Image
        replace_mapping = {
            "\nOptions:": "\nChoices:",
            "Please select the correct answer from the options above.": "Answer with the letter.",
        }
        prompt, images = "<|im_start|>User:", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += "<image>"
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction
        prompt += "<end_of_utterance>\nAssistant: Answer:"
        images = (
            [images]
            if isinstance(images, Image.Image)
            else images
        )
        inputs = image_processor(
            text=prompt, images=images, return_tensors="pt"
        ).to("cuda")
        return inputs, None
    elif 'hf_v16' in str(tokenizer):
        content, images = [], []
        for msg in message:
            if msg["type"] == "text":
                content.append({"type": msg["type"], "text": msg["value"]})
            elif msg["type"] == "image":
                content.append({"type": "image"})
                images.append(Image.open(msg["value"]).convert("RGB"))
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]
        prompt = image_processor.apply_chat_template(
        conversation, add_generation_prompt=True
        )
        inputs = image_processor(prompt, images, return_tensors="pt").to(
                    "cuda", torch.float16)
        return inputs, None
    else:
        print(f"tokenizer is {tokenizer}")
    system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
    def concat_tilist(message):
        text, images = "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                text += " <image> "
                images.append(item["value"])
        return text, images
    
    content, images = concat_tilist(message)
    images = [Image.open(s).convert("RGB") for s in images]
    if images:
        image_tensor = process_images(images, image_processor, model.config).to(
            utils.get_dev(), dtype=torch.float16
        )
        image_sizes = [img.size for img in images] # for llava 1.6// one vision
    else:
        image_tensor = None
        image_sizes = None
    prompt = system_prompt + "USER: " + content + " ASSISTANT: "
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(utils.get_dev())
    return input_ids, (image_tensor, image_sizes)


def message_to_prompt_train(message, image_processor, model, tokenizer, label_mode="q-a", label_return=False):
    from llava.mm_utils import (
        process_images,
        tokenizer_image_token,
        KeywordsStoppingCriteria,
    )
    from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
    from PIL import Image
    if tokenizer is None: # later may add model_type to decide
        from transformers.image_utils import load_image
        from PIL import Image
        replace_mapping = {
            "\nOptions:": "\nChoices:",
            "Please select the correct answer from the options above.": "Answer with the letter.",
        }
        prompt, images = "<|im_start|>User:", []
        anw = ""
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += "<image>"
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction
            elif msg["type"] == "textanw":
                anw += msg["value"]
        prompt += "<end_of_utterance>\nAssistant: Answer:"
        images = (
            [images]
            if isinstance(images, Image.Image)
            else images
        )
        if label_mode == 'q-a':
            inputs = image_processor(text=prompt, images=images, return_tensors="pt").to("cuda") # IMAGE TOKEN + Question
            answer = image_processor(text=anw, return_tensors="pt").to("cuda") # Answer
            return inputs, None, answer # to accommodate llava input?
        elif label_mode == 'qa-qa':
            question_prompt = prompt
            prompt += anw

            inputs = image_processor(text=prompt, images=images, return_tensors="pt").to("cuda", torch.float16) # image + question + answer
            question_token = image_processor(text=question_prompt, images=images, return_tensors="pt").to("cuda", torch.float16) # image + question
            
            question_ids = question_token['input_ids'].clone()
            input_ids = inputs['input_ids'].clone()

            labels = inputs['input_ids'].clone()

            # Locate question part within the full input
            # Method 1: if question always starts at the beginning, use question length directly
            question_length = question_ids.size(1)
            labels[:, :question_length] = IGNORE_INDEX # mask image and question, keep all answer with "question_length-1"
            # breakpoint()
            if image_processor.tokenizer.pad_token_id is not None:
                labels[labels == image_processor.tokenizer.pad_token_id] = -100

            inputs['input_ids'] = inputs['input_ids'][:,:-1]
            labels = labels[:,1:]
            # breakpoint()
            return inputs, None, labels

    elif tokenizer == 'hf_v16':
        if label_mode == 'q-a':
            content, images = [], []
            anw = []
            for msg in message:
                if msg["type"] == "text":
                    content.append({"type": msg["type"], "text": msg["value"]})
                elif msg["type"] == "textanw":
                    anw.append({"type": "text", "text": msg["value"]})
                else:
                    content.append({"type": "image"})
                    images.append(Image.open(msg["value"]).convert("RGB"))

            conversation = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            answer = [
                {
                    "role": "user",
                    "content": anw,
                }
            ]
            prompt = image_processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = image_processor(prompt, images, return_tensors="pt").to("cuda", torch.float16) # IMAGE TOKEN + Question  
            
            answer = image_processor.apply_chat_template(
                answer, # add_generation_prompt=True # whether here add generation prompt?
            )
            answer = image_processor(answer, return_tensors="pt").to("cuda", torch.float16) # Answer

            input_ids = inputs.get('input_ids')
            output_ids = answer.get('input_ids')

            final_output_ids = insert_ignore_index_after_prompt(input_ids[0],output_ids[0], image_token_id=32000, ignore_index=-100) # This image_token_id is for LLaVANextProcessor
            # Create a new output_ids tensor matching final_output_ids size
            new_output_ids = torch.full((output_ids.size(0), final_output_ids.size(0)), 
                                            fill_value=-100, 
                                            dtype=output_ids.dtype, 
                                            device=output_ids.device)
            # Assign final_output_ids to the first sample
            new_output_ids[0] = final_output_ids
            # Replace original output_ids
            output_ids = new_output_ids # labels

            return inputs, None, output_ids
        elif label_mode == 'qa-qa':
            content, images = [], []
            anw = []
            for msg in message:
                if msg["type"] == "text":
                    # Remove specific prompt text
                    msg_clean = msg["value"].replace("\nAnswer with the option's letter from the given choices directly.", "")
                    content.append({"type": msg["type"], "text": msg_clean})
                elif msg["type"] == "textanw":
                    anw.append({"type": "text", "text": msg["value"]})
                else:
                    content.append({"type": "image"})
                    images.append(Image.open(msg["value"]).convert("RGB"))

            conversation = [
                {
                    "role": "user",
                    "content": content,
                },
                {
                    "role": "assistant",
                    "content": anw,
                }
            ]
            question = [
                {
                    "role": "user",
                    "content": content,
                },
                {
                    "role": "assistant",
                    "content": None,
                }
            ]
            prompt = image_processor.apply_chat_template(conversation, add_generation_prompt=False)
            inputs = image_processor(prompt, images, return_tensors="pt").to("cuda", torch.float16) # image + question + answer
            question_prompt = image_processor.apply_chat_template(question, add_generation_prompt=False)
            question_token = image_processor(question_prompt, images, return_tensors="pt").to("cuda", torch.float16) # image + question
            
            question_ids = question_token['input_ids'].clone()
            input_ids = inputs['input_ids'].clone()

            labels = inputs['input_ids'].clone()

            # Locate question part within the full input
            # Method 1: if question always starts at the beginning, use question length directly
            question_length = question_ids.size(1)
            labels[:, :question_length-1] = IGNORE_INDEX # mask image and question, keep all answer with "question_length-1"

            if image_processor.tokenizer.pad_token_id is not None:
                labels[labels == image_processor.tokenizer.pad_token_id] = -100

            # inputs["labels"] = labels
            # breakpoint()
            inputs['input_ids'] = inputs['input_ids'][:,:-1]
            labels = labels[:,1:]
            return inputs, None, labels
    
    elif tokenizer == 'qwen':
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err
        
        content = []
        anw = []
        for s in message:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                item['max_pixels'] = 16384 * 28 * 28
                item['min_pixels'] = 1280 * 28 * 28
                content.append(item)
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
                content.append(item)
            elif s['type'] == 'textanw':
                anw.append({"type": "text", "text": s['value']})
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
        
        messages = []
        messages.append({'role': 'user', 'content': content})
        input_text = image_processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])
        inputs = image_processor(text=input_text, images=images, videos=videos, padding=True, return_tensors='pt').to("cuda")
        
        # Process answer
        answer = None
        if anw:
            anwser = [
                {
                    "role": "assistant",
                    "content": anw,
                }
             ]
            anwser_text = image_processor.apply_chat_template([anwser], tokenize=False, add_generation_prompt=True)
            answer = image_processor(text=anwser_text, return_tensors="pt").to("cuda")
        return inputs, None, answer
    
    elif tokenizer == 'intern':
        from intern_utils import load_image
        image_num = len([x for x in message if x['type'] == 'image'])
        max_num = 6
        total_max_num = 64
        max_num = max(1, min(max_num, total_max_num // image_num))
        device = utils.get_dev()
        dataset = None
        prompt = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        label = ''.join([x['value'] for x in message if x['type'] == 'textanw'])
        # prompt = reorganize_prompt(message, image_num, dataset) # we do not need dataset here?
        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            num_patches_list, pixel_values_list = [], []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU'], dataset)
                curr_pixel_values = load_image(
                    file_name, max_num=max_num, upscale=upscale_flag).to(device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            upscale_flag = dataset is not None and listinstr(['MMMU'], dataset)
            pixel_values = load_image(
                image_path, max_num=max_num, upscale=upscale_flag).to(device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []
        kwargs_default = dict(
            do_sample = False,
            max_new_tokens = 512,
            temperature = 0.6,
            top_p = 0.95
        )
        # adding things here?
        if label_return:
            return {'pixel_values': pixel_values, 'num_patches_list': num_patches_list, 'question': prompt, 'generation_config': kwargs_default, 'label': label}
        else:
            return {'pixel_values': pixel_values, 'num_patches_list': num_patches_list, 'question': prompt, 'generation_config': kwargs_default}
    else:
        print(f"tokenizer is {tokenizer}")
    system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
    def concat_tilist(message):
        anw, text, images = "", "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                text += " <image> "
                images.append(item["value"])
            elif item["type"] == "textanw":
                anw += item["value"]
        return text, images, anw
    
    content, images, anw = concat_tilist(message)
    images = [Image.open(s).convert("RGB") for s in images]
    if images:
        image_tensor = process_images(images, image_processor, model.config).to(
            utils.get_dev(), dtype=torch.float16
        )
        image_sizes = [img.size for img in images] # for llava 1.6// one vision
    else:
        image_tensor = None
        image_sizes = None
    if label_mode == 'q-a':
        prompt = system_prompt + "USER: " + content + " ASSISTANT: "
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(utils.get_dev())
        answer = tokenizer_image_token(anw, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(utils.get_dev())
        return input_ids, (image_tensor, image_sizes), answer
    elif label_mode == 'qa-qa':
        prompt = system_prompt + "USER: " + content + " ASSISTANT: " + anw
        question_prompt = system_prompt + "USER: " + content + " ASSISTANT: "
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(utils.get_dev())
        question_ids = tokenizer_image_token(question_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(utils.get_dev())
        answer = input_ids.clone()
        question_length = question_ids.size(1)
        del question_ids
        answer[:, :question_length-1] = IGNORE_INDEX
        if tokenizer.pad_token_id is not None:
            answer[answer == tokenizer.pad_token_id] = -100
        input_ids = input_ids[:,:-1]
        answer = answer[:,1:]
        return input_ids, (image_tensor, image_sizes), answer


def message_to_promptsmolvlm(message, image_processor, model, tokenizer):
    if tokenizer is None:
        from transformers.image_utils import load_image
        from PIL import Image
        replace_mapping = {
            "\nOptions:": "\nChoices:",
            "Please select the correct answer from the options above.": "Answer with the letter.",
        }
        prompt, images = "<|im_start|>User:", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += "<image>"
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction
        prompt += "<end_of_utterance>\nAssistant: Answer:"
        images = (
            [images]
            if isinstance(images, Image.Image)
            else images
        )
        inputs = image_processor(
            text=prompt, images=images, return_tensors="pt"
        ).to("cuda")
        return inputs

def message_to_prompt_trainsmolvlm(message, image_processor, model, tokenizer):
    if tokenizer is None:
        from transformers.image_utils import load_image
        from PIL import Image
        replace_mapping = {
            "\nOptions:": "\nChoices:",
            "Please select the correct answer from the options above.": "Answer with the letter.",
        }
        prompt, images = "<|im_start|>User:", []
        anw = ""
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += "<image>"
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction
            elif msg["type"] == "textanw":
                anw += msg["value"]
        prompt += "<end_of_utterance>\nAssistant: Answer:"
        images = (
            [images]
            if isinstance(images, Image.Image)
            else images
        )
        inputs = image_processor(
            text=prompt, images=images, return_tensors="pt"
        ).to("cuda")
        answer = image_processor(text=anw, return_tensors="pt").to("cuda")
        return inputs, None,  answer # to accommodate llava input?


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')

@torch.no_grad()
def rtn_fwrdsmovlmvit(model, dev, args, start_id=0):
    '''
    From GPTQ repo 
    TODO: Make this fuction general to support both OPT and LLaMA models
    '''
    assert args.w_groupsize ==-1, "Groupsize not supported in RTN!"
    layers = model.model.vision_model.encoder.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(start_id, len(layers)), desc="(RtN vit Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(layer,
                                            layers=[torch.nn.Linear])

        for name in subset:
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and 'fc2' in name:
                layer_weight_bits = 8

            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(
                next(iter(layer.parameters())).dtype)
            quantizers['model.vitlayers.%d.%s' % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer
    del layers
    utils.cleanup_memory(verbos=True)
    return quantizers


@torch.no_grad()
def rtn_fwrdsmovlmmm(model, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    assert args.w_groupsize ==-1, "Groupsize not supported in RTN!"
    
    torch.cuda.empty_cache()

    quantizers = {}


    layer = model.model.connector.to(dev)

    subset = quant_utils.find_qlayers(layer,
                                        layers=[torch.nn.Linear])

    for name in subset:
        layer_weight_bits = args.w_bits
        if 'lm_head' in name:
            layer_weight_bits = 16
            continue
        if args.int8_down_proj and 'down_proj' in name:
            layer_weight_bits = 8

        quantizer = quant_utils.WeightQuantizer()
        quantizer.configure(
            layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
        )
        W = subset[name].weight.data
        dtype = W.dtype
        quantizer.find_params(W)
        subset[name].weight.data = quantizer.quantize(W).to(
            dtype)
        quantizers['model.connector.%s' % (name)] = quantizer.cpu()
    model.model.connector = layer.cpu()
    torch.cuda.empty_cache()
    del layer
            
    utils.cleanup_memory(verbos=True)
    return quantizers


    