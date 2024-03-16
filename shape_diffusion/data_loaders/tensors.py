import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
    return mask

def lengths_to_mask2(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    mask_2 = torch.zeros(mask.shape)
    mask_2[mask==False] = 1.0
    return mask_2.float()
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def collate_posed_handle(batch):
    # batch[(T, 40, 3)]
    max_length = 196
    canvas_lst = []
    for handle in batch:
        length = handle.shape[0]
        canvas = torch.zeros((1, max_length, handle.shape[1], handle.shape[2]))
        if length>=max_length:
            canvas[0, :, :, :] = handle[:max_length, :, :]
            canvas_lst.append(canvas)
        else:
            repeat_key = int(max_length / length)
            mod = max_length % length
            handle_rp = handle.repeat((repeat_key, 1, 1))
            canvas[0, :repeat_key*length, :, :] = handle_rp[:, :, :]
            canvas[0, repeat_key*length:, :, :] = handle[:mod, :, :]
            canvas_lst.append(canvas)
    
    posed_handle = torch.cat(canvas_lst, dim=0)
    posed_handle = posed_handle.reshape((posed_handle.shape[0], posed_handle.shape[1], -1))
    return posed_handle


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1])       # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        textbatch_length = torch.as_tensor([len(b.split(" ")) for b in textbatch])         # +1 for additional t condition
        text_mask = lengths_to_mask(textbatch_length, 30)
        cond['y'].update({'text': textbatch})
        cond['y'].update({'text_mask': text_mask})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text'] for b in notnone_batches]
        cond['y'].update({'action_text': action_text})
    
    if 'char_feature' in notnone_batches[0]:
        char_feature = [b['char_feature'] for b in notnone_batches]
        char_feature = torch.cat(char_feature, dim=0)                        # bs 256
        cond['y'].update({'char_feature': char_feature})
    
    if 'char_name' in notnone_batches[0]:
        char_name = [b['char_name'] for b in notnone_batches]
        cond['y'].update({'char_name': char_name})
    
    if 'char_handle' in notnone_batches[0]:
        char_handle = [b['char_handle'] for b in notnone_batches]
        char_handle = torch.cat(char_handle, dim=0)                        # bs 40 3
        cond['y'].update({'char_handle': char_handle})
    
    if 'theta' in notnone_batches[0]:
        thetabatch = collate_tensors([b['theta'] for b in notnone_batches])

    if 'focus_verts' in notnone_batches[0]:
        focus_verts = [b['focus_verts'][None,] for b in notnone_batches]
        focus_verts = torch.cat(focus_verts, dim=0)                        # bs 50 3
        cond['y'].update({'focus_verts': focus_verts})
    
    if 'focus_skin' in notnone_batches[0]:
        focus_score = [b['focus_skin'] for b in notnone_batches]
        focus_score = torch.cat(focus_score, dim=0)                         # bs 50 30
        cond['y'].update({'focus_skin': focus_score})
    

    if "posed_handle" and "posed_verts" in notnone_batches[0]:
        posed_handle_batch = collate_posed_handle([b['posed_handle'] for b in notnone_batches])
        posed_verts_batch = collate_posed_handle([b['posed_verts'] for b in notnone_batches])
        return motion, cond, thetabatch, posed_handle_batch, posed_verts_batch
    else:
        return motion, cond, thetabatch

# def collate(batch):
#     notnone_batches = [b for b in batch if b is not None]
#     databatch = [b['inp'] for b in notnone_batches]
#     if 'lengths' in notnone_batches[0]:
#         lenbatch = [b['lengths'] for b in notnone_batches]
#     else:
#         lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


#     databatchTensor = collate_tensors(databatch)
#     lenbatchTensor = torch.as_tensor(lenbatch)
#     maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

#     motion = databatchTensor
#     cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

#     if 'text' in notnone_batches[0]:
#         textbatch = [b['text'] for b in notnone_batches]
#         cond['y'].update({'text': textbatch})

#     if 'tokens' in notnone_batches[0]:
#         textbatch = [b['tokens'] for b in notnone_batches]
#         cond['y'].update({'tokens': textbatch})

#     if 'action' in notnone_batches[0]:
#         actionbatch = [b['action'] for b in notnone_batches]
#         cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

#     # collate action textual names
#     if 'action_text' in notnone_batches[0]:
#         action_text = [b['action_text']for b in notnone_batches]
#         cond['y'].update({'action_text': action_text})

#     return motion, cond
    

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'theta': torch.tensor(b[5]).float(),                      # T 22*3
        'text': b[2], #b[0]['caption']
        'tokens': b[7],
        'lengths': b[6],
    } for b in batch]
    return collate(adapted_batch)


def t6d_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'theta': torch.tensor(b[5]).float(),                      # T 22*3
        'text': b[2], #b[0]['caption']
        'tokens': b[7],
        'lengths': b[6],
        'char_feature': torch.tensor(b[9]).float(),         # 1 256
        'char_handle': torch.tensor(b[10]).float(),
        'char_name': b[8],
        'posed_handle': torch.tensor(b[11]).float(),         # T 40 3
        'focus_verts':torch.tensor(b[12]).float(),           # 50 3       
        'focus_skin':torch.tensor(b[13]).float(),           # 50 30
        'posed_verts':torch.tensor(b[14]).float()            # T 50 3
    } for b in batch]
    return collate(adapted_batch)


