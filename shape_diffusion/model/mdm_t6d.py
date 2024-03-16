import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.trans2mesh import Trans2Mesh
from transformers import AutoModel, AutoTokenizer


class MDM_T6d(nn.Module):
    def __init__(self, modeltype, njoints, num_actions, translation, smpl_path, part_path,
                 latent_dim=512, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, dataset='trans', clip_dim=768,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.num_actions = num_actions
        self.dataset = dataset
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats_rot = (self.njoints+1) * 6
        self.input_feats_ofs = (self.njoints+1) * 3

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.input_feats_rot, self.input_feats_ofs, self.latent_dim, num_joints=self.njoints+1)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

    
        if self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        else:
            raise ValueError('Please choose correct architecture [trans_dec]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.load_and_freeze_hdclip(clip_version)
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        self.output_process = OutputProcess(self.input_feats_rot, self.input_feats_ofs, self.latent_dim, self.njoints+1)

        self.mesh_encoder = MeshFeatureEmbedder(self.latent_dim)

        self.trans2mesh = Trans2Mesh(smpl_path, part_path)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
    
    def load_and_freeze_hdclip(self, clip_path):
        self.tokenizer = AutoTokenizer.from_pretrained(clip_path)
        self.text_model = AutoModel.from_pretrained(clip_path)
        self.text_model.training = False
        self.tokenizer.training = False
        for p in self.text_model.parameters():
            p.requires_grad = False

    def mask_cond(self, cond, force_mask=False):
        seq_len, bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
    
    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit', 't6d_mixrig'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    
    def encode_text_hdclip(self, raw_text):
        device = next(self.parameters()).device
        max_text_len = None
        default_context_length =  30                      # 77
        if max_text_len is not None:
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            text_inputs = self.tokenizer(
                raw_text,
                padding="max_length",
                truncation=True,
                max_length=context_length,
                return_tensors="pt",
            )
        else:
            text_inputs = self.tokenizer(
                raw_text,
                padding="max_length",
                truncation=True,
                max_length=default_context_length,
                return_tensors="pt",
            )
        text_input_ids = text_inputs.input_ids.to(device)
        text_mask = text_inputs.attention_mask.to(device)
        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            text_input_ids = text_input_ids[:, :self.tokenizer.model_max_length]
        text_embeddings = self.text_model.text_model(text_input_ids).last_hidden_state

        return text_embeddings, text_mask          # bs seq_len d
    
    def get_tgt_mask(self, size):
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        return mask

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        device = x.device
        emb_t = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text, text_mask = self.encode_text_hdclip(y['text'])
            enc_text = enc_text.permute(1,0,2).contiguous()   # seq_len, bs, d
            emb_text = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
            emb_mesh = self.mesh_encoder(self.mask_cond(y['char_feature'][None, ].to(device), force_mask=force_mask))

            emb = torch.cat([emb_t, emb_text], dim=0)
            emb = torch.cat([emb_mesh, emb], dim=0)              # seq_len+2, bs, d

        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        x = self.input_process(x)
        xseq = self.sequence_pos_encoder(x)

        x_mask = y['mask'].float().to(device)
        t_mask = torch.zeros((bs, 2)).to(device)
        emb_mask = torch.cat([t_mask ,(1.0-text_mask).to(device)], dim=1)

        tgt_mask = ~(torch.triu(torch.ones(nframes, nframes)) == 1).transpose(0, 1).to(device)

        output = self.seqTransDecoder(tgt=xseq, memory=emb, tgt_mask=tgt_mask, tgt_key_padding_mask=x_mask, memory_key_padding_mask=emb_mask)    # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)

class MeshFeatureEmbedder(nn.Module):
    def __init__(self, latent_dim, input_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.mesh_emb = nn.Linear(self.input_dim, self.latent_dim)
    
    def forward(self, mesh_feature):
        # mesh_feature (bs, 256)
        return self.mesh_emb(mesh_feature)


class InputProcess(nn.Module):
    def __init__(self, input_feats_rot, input_feats_ofs, latent_dim, num_joints=31):
        super().__init__()
        self.input_feats_rot = input_feats_rot
        self.input_feats_ofs = input_feats_ofs
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.rotEmbedding = nn.Linear(self.input_feats_rot, self.latent_dim//2)
        self.ofsEmbedding = nn.Linear(self.input_feats_ofs, self.latent_dim//2)

    def forward(self, x):
        # bs 287 1 
        bs, njoints_nfeatures, _, nframes = x.shape
        x = x.reshape(bs, self.num_joints, -1, nframes)
        x_rot = x[:, :, 3:, :]     # B N 6 T
        x_ofs = x[:, :, :3, :]     # B N 3 T

        x_rot = x_rot.permute((3, 0, 1, 2)).contiguous().reshape(nframes, bs, self.num_joints*6)
        x_ofs = x_ofs.permute((3, 0, 1, 2)).contiguous().reshape(nframes, bs, self.num_joints*3)

        eb_rot = self.rotEmbedding(x_rot)
        eb_ofs = self.ofsEmbedding(x_ofs)

        return torch.cat((eb_ofs, eb_rot), dim=2)   # T B D
    
class Attention(nn.Module):
    def __init__(self, dim, out_dim, heads = 1, dropout = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim*3*heads, bias = True)
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)
        
        self.nn1 = nn.Linear(dim, out_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)        
        self.do1 = nn.Dropout(dropout)
        

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x) #gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        # q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # split into multi head attentions

        qkv = qkv.view(b, n, 3, h, -1).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0,:,:,:,:], qkv[1,:,:,:,:], qkv[2,:,:,:,:]

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1) #follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v) #product of v times whatever inside softmax
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, -1)
        # out = rearrange(out, 'b h n d -> b n (h d)') #concat heads into one matrix, ready for next encoder block
        out =  self.nn1(out)
        out = self.do1(out)

        return out


class OutputProcess(nn.Module):
    def __init__(self, input_feats_rot, input_feats_ofs, latent_dim, njoints):
        super().__init__()
        self.input_feats_rot = input_feats_rot
        self.input_feats_ofs = input_feats_ofs
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.rotFinal = nn.Linear(self.latent_dim, self.input_feats_rot)
        self.ofsFinal = nn.Linear(self.latent_dim, self.input_feats_ofs)
        self.ofsModi = nn.Linear(self.latent_dim, self.input_feats_ofs)
        
    def forward(self, output):
        nframes, bs, _ = output.shape
        output_rot = self.rotFinal(output)
        output_ofs = self.ofsFinal(output)

        output_mod = self.ofsModi(output)                     # B T D

        output_rot = output_rot.reshape((nframes, bs, self.njoints, -1))    # T B N D
        output_ofs = output_ofs.reshape((nframes, bs, self.njoints, -1))    # T B N D
        output_mod = output_mod.reshape((nframes, bs, self.njoints, -1))

        output = torch.cat((output_ofs, output_rot), dim=3)     # T B N 3+6
        output = output.permute(1, 2, 3, 0).contiguous()
        output_mod = output_mod.permute(1, 2, 3, 0).contiguous()
        return output, output_mod


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output