import torch
import numpy as np
from kornia.geometry.conversions import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix, QuaternionCoeffOrder
wxyz = QuaternionCoeffOrder.WXYZ
from utils.geometry import fps, arap_loss
from torch_scatter import scatter_sum, scatter_mean
from torch.nn.functional import kl_div
import math


def distChamfer(a, b, norm="L2"):
    """
    a: (B, N1, 3)
    b: (B, N2, 3)
    """
    x = a[:, :, None]
    y = b[:, None]
    if norm == "L1":
        dist = torch.abs(x - y)
    else:
        dist = torch.square(x - y)
    dist = torch.sum(dist, -1)
    return torch.mean(torch.min(dist, 2)[0]) + torch.mean(torch.min(dist, 1)[0])


def selfChamfer(a, norm="L2"):
    """
    a: (B, N1, 3)
    """
    N = a.shape[1]
    x = a[:, :, None]
    y = a[:, None]
    if norm == "L1":
        dist = torch.abs(x - y)
    else:
        dist = torch.square(x - y)
    dist = torch.sum(dist, -1)
    for i in range(N):
        dist[:, i, i] += 10000
    closest1 = torch.min(dist, 2)[0]
    closest2 = torch.min(dist, 1)[0]
    closest1[closest1>0.05] = 0
    closest2[closest2>0.05] = 0
    return torch.mean(closest1) + torch.mean(closest2)



def sw_loss(region_score, gt_weights, batch):
    """
    use GT skinning weights as supervision
    region_score: (B*V, K)
    gt_weights: list (V, K2) length: B
    """
    B = len(gt_weights)
    K = region_score.shape[1]
    all_loss = []
    for i in range(B):
        scores = region_score[batch==i] + 1e-6
        seg = torch.argmax(scores, 1).detach().cpu().numpy()
        gt_weight = gt_weights[i]
        if not isinstance(gt_weight, np.ndarray):
            gt_weight = np.array(gt_weight.todense())
        gt_seg = np.argmax(gt_weight, 1)
        gt_valid = np.where(np.max(gt_weight, 1) > 0.9)[0]
        p1, p2 = [], []
        for j in range(K):
            seg_idxs = np.where(seg == j)[0]
            np.random.shuffle(seg_idxs)
            if len(seg_idxs) < 2:
                continue
            half_num = len(seg_idxs) // 2
            p1.append(seg_idxs[:half_num])
            p2.append(seg_idxs[half_num: 2*half_num])
        p1 = np.concatenate(p1, 0)
        p2 = np.concatenate(p2, 0)
        valid_pidx = np.logical_and(np.isin(p1, gt_valid), np.isin(p2, gt_valid))
        p1 = p1[valid_pidx]
        p2 = p2[valid_pidx]
        same = gt_seg[p1] == gt_seg[p2]
        diff = np.logical_not(same)
        same = np.where(same)[0]
        diff = np.where(diff)[0]
        p1_score = scores[p1]
        p2_score = scores[p2]
        
        loss = kl_div(p1_score[same].log(), p2_score[same]) - kl_div(p1_score[diff].log(), p2_score[diff])

        # exp to make the loss non-neg
        # loss = torch.exp(kl_div(p1_score[same].log(), p2_score[same])) + torch.exp(- kl_div(p1_score[diff].log(), p2_score[diff])) 

        if torch.isnan(loss):
            continue
        all_loss.append(loss)
    if len(all_loss) == 0:
        return torch.tensor(0).float().to(region_score.device)
    else:
        return torch.mean(torch.stack(all_loss))

def root_sw_loss(region_score, gt_weights, batch):
    """
    region_score: (B*V, K)
    gt_weights: list (V, K2) length: B
    """
    B = len(gt_weights)
    K = region_score.shape[1]
    all_loss = []
    for i in range(B):
        scores = region_score[batch==i] + 1e-6
        scores_root = scores[:, 0]                     # V
        gt_root = gt_weights[i][:, 0]                  # V
        loss = kl_div(scores_root.log(), torch.from_numpy(gt_root).float().to(scores_root.device))
        if torch.isnan(loss):
            continue
        all_loss.append(loss)

    if len(all_loss) == 0:
        return torch.tensor(0).float().to(region_score.device)
    else:
        return torch.mean(torch.stack(all_loss))


def center_sw_loss(gt_weights, hm, v0, batch):
    """
    gt_weights: list (V, K2) length: B
    hm: (B*V, K)
    v0: (B*V, 3)
    """
    B = len(gt_weights)
    hd0 = scatter_sum(hm[:, :, None] * v0[:, None], batch, dim=0) # (B, 40, 3)
    hd0_root = hd0[:, 0, :]

    hdgt_root_lst = []
    for i in range(B):
        gt_weights_i = torch.from_numpy(gt_weights[i]).float().to(hd0_root.device)   # (V, K2)
        gt_score_i = gt_weights_i / torch.sum(gt_weights_i, dim=0, keepdim=True)
        v0_i= v0[batch==i]
        hdgt = torch.sum(gt_score_i[:, :, None] * v0_i[:, None], dim=0) # (K2, 3)
        hdgt_root = hdgt[[0], :]
        hdgt_root_lst.append(hdgt_root)
    
    hdgt_root = torch.cat(hdgt_root_lst, dim=0)

    return torch.sum(torch.mean((hdgt_root-hd0_root)**2, dim=0))


def center_sw_loss2(hm, v0, batch):
    """
    hm: (B*V, K)
    v0: (B*V, 3)
    """
    hd0 = scatter_sum(hm[:, :, None] * v0[:, None], batch, dim=0) # (B, 40, 3)
    hd0_root = hd0[:, 0, :]
    v0_root = scatter_mean(v0, batch, dim=0) # (B, 3)

    return torch.sum(torch.mean((v0_root-hd0_root)**2, dim=0))


def quaternion_loss(q1, q2):
    l1 = (q1 - q2).abs().mean(-1)
    l2 = (q1 + q2).abs().mean(-1)
    mask = (l1 < l2).float()
    loss = l1 * mask + l2 * (1-mask)
    return torch.mean(loss)


def transformation_loss(transformation, hm, region_score, batch, v0, v1, criterion):
    hd0 = scatter_sum(hm[:, :, None] * v0[:, None], batch, dim=0)
    hd1 = scatter_sum(hm[:, :, None] * v1[:, None], batch, dim=0)  # (B, 40, 3)
    r_weight = 0.5
    with torch.no_grad():
        B, K, _ = hd0.shape
        seg = torch.max(region_score, 1)[1]  # (B*V,)
        R_batch = []
        for i in range(B):
            center0 = hd0[i]  # (40, 3)
            center1 = hd1[i]  # (40, 3)
            seg_single = seg[batch==i]  # (V,)
            v0_single = v0[batch==i]  # (V, 3)
            v1_single = v1[batch==i]  # (V, 3)
            seg_max = torch.max(seg_single) + 1

            scatter_center0 = center0[seg_single]
            scatter_center1 = center1[seg_single]

            v0_single = v0_single - scatter_center0
            v1_single = v1_single - scatter_center1

            M = torch.bmm(v0_single[:, :, None], v1_single[:, None])  # (V, 3, 3)
            M_sum = scatter_sum(M, seg_single, dim=0)  # (40, 3, 3)
            try:
                U, S, Vh = torch.linalg.svd(M_sum)
                # u, s, vh = np.linalg.svd(M_sum.cpu().numpy())
                # U = torch.from_numpy(u).float().to(M_sum.device)
                # S = torch.from_numpy(s).float().to(M_sum.device)
                # Vh = torch.from_numpy(vh).float().to(M_sum.device)
            except RuntimeError as e:
                print(e)
                r_weight = 0
                R_batch.append(torch.eye(3)[None].repeat(K, 1, 1).float().to(M_sum.device))
                continue
            R_temp = torch.bmm(U, Vh).permute(0, 2, 1)  # (40, 3, 3)
            R = torch.eye(3)[None].repeat(K, 1, 1).float().to(R_temp.device)
            R[:seg_max] = R_temp
            R = rotation_matrix_to_quaternion(R, order=wxyz)
            R_batch.append(R)

        R_batch = torch.stack(R_batch, 0)  # (B, 40, 3, 3)

        # R_batch = R_batch.detach().cpu().numpy()
        # Rs = R_batch.reshape([-1, 3, 3])
        # R_batch = []
        # for R in Rs:
        #     R_batch.append(cv2.Rodrigues(R)[0])
        # R_batch = np.stack(R_batch, 0)
        # R_batch = R_batch.reshape([B, K, 3])
        # R_batch = torch.from_numpy(R_batch).float().to(hd0.device)

        t_batch = hd1 - hd0
    pred_t = transformation[:, :, :3]
    pred_R = transformation[:, :, 3:]
    if r_weight == 0:
        return criterion(t_batch, pred_t)
    else:
        return quaternion_loss(R_batch, pred_R) * r_weight + criterion(t_batch, pred_t)


def get_transformation(hm, region_score, batch, v0, v1):
    with torch.no_grad():
        # hd is the weighted center of ecah part
        hd0 = scatter_sum(hm[:, :, None] * v0[:, None], batch, dim=0)
        hd1 = scatter_sum(hm[:, :, None] * v1[:, None], batch, dim=0)  # (B, 40, 3)

        B, K, _ = hd0.shape                   # 1 40 3
        seg = torch.max(region_score, 1)[1]  # (B*V,)

        R_batch = []
        for i in range(B):
            center0 = hd0[i]  # (40, 3)
            center1 = hd1[i]  # (40, 3)
            seg_single = seg[batch==i]  # (V,)   indicates each vertex belongs to which part
            v0_single = v0[batch==i]  # (V, 3)
            v1_single = v1[batch==i]  # (V, 3)
            seg_max = torch.max(seg_single) + 1

            scatter_center0 = center0[seg_single]
            scatter_center1 = center1[seg_single]

            v0_single = v0_single - scatter_center0
            v1_single = v1_single - scatter_center1

            M = torch.bmm(v0_single[:, :, None], v1_single[:, None])  # (V, 3, 3)
            M_sum = scatter_sum(M, seg_single, dim=0)  # (40, 3, 3)
            U, S, Vh = torch.linalg.svd(M_sum)
            R_temp = torch.bmm(U, Vh).permute(0, 2, 1)  # (40, 3, 3)
            R = torch.eye(3)[None].repeat(K, 1, 1).float().to(R_temp.device)
            R[:seg_max] = R_temp
            R = rotation_matrix_to_quaternion(R, order=wxyz)
            R_batch.append(R)

        R_batch = torch.stack(R_batch, 0)  # (B, 40, 4)
        t_batch = hd1 - hd0                # (B, 40, 3)
    return torch.cat((t_batch, R_batch), 2)

def get_transformation_root(hm, region_score, batch, v0, v1):
    with torch.no_grad():
    # hd is the weighted center of ecah part
        hd0 = scatter_sum(hm[:, :, None] * v0[:, None], batch, dim=0)
        hd1 = scatter_sum(hm[:, :, None] * v1[:, None], batch, dim=0)  # (B, 40, 3)

        B, K, _ = hd0.shape                   # 1 40 3

        hd0_root = hd0[:,[0],:]  # B, 1, 3
        hd1_root = hd1[:,[0],:]

        seg = torch.max(region_score, 1)[1]  # (B*V,)

        R_batch = []
        for i in range(B):
            # center0 = hd0[i]  # (40, 3)
            # center1 = hd1[i]  # (40, 3)
            seg_single = seg[batch==i]  # (V,)   indicates each vertex belongs to which part
            v0_single = v0[batch==i]  # (V, 3)
            v1_single = v1[batch==i]  # (V, 3)
            seg_max = torch.max(seg_single) + 1

            # scatter_center0 = center0[seg_single]
            # scatter_center1 = center1[seg_single]

            hd0_root_single = hd0_root[i]
            hd1_root_single = hd1_root[i]

            v0_single = v0_single - hd0_root_single              # move the root to (0,0,0)
            v1_single = v1_single - hd1_root_single

            M = torch.bmm(v0_single[:, :, None], v1_single[:, None])  # (V, 3, 3)
            M_sum = scatter_sum(M, seg_single, dim=0)  # (40, 3, 3)
            U, S, Vh = torch.linalg.svd(M_sum)
            R_temp = torch.bmm(U, Vh).permute(0, 2, 1)  # (40, 3, 3)
            R = torch.eye(3)[None].repeat(K, 1, 1).float().to(R_temp.device)
            R[:seg_max] = R_temp
            R = rotation_matrix_to_quaternion(R, order=wxyz)
            R_batch.append(R)

        R_batch = torch.stack(R_batch, 0)  # (B, 40, 4)
        t_batch = hd1_root - hd0_root               # (B, 1, 3)
        t_batch = t_batch.repeat(1,K,1)
    return torch.cat((t_batch, R_batch), 2)


def repr6d2mat(repr):
    x = repr[..., :3]
    y = repr[..., 3:]
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    z = np.cross(x, y)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    res = [x, y, z]
    res = [np.expand_dims(v, axis=-2) for v in res]
    mat = np.concatenate(res, axis=-2)
    return mat


def repr6d2mat_tensor(repr):
    x = repr[..., :3]
    y = repr[..., 3:]
    x = x / x.norm(dim=-1, keepdim=True)
    z = torch.cross(x, y)
    z = z / z.norm(dim=-1, keepdim=True)
    y = torch.cross(z, x)
    res = [x, y, z]
    res = [v.unsqueeze(-2) for v in res]
    mat = torch.cat(res, dim=-2)
    return mat


def handle2mesh(transformation, handle_pos, region_score, batch, v0):
    """
    use per-part trans+rot to reconstruct the mesh
    transformation: (B, 40, 3+4)
    handle_pos: handle position of T-pose
    handle_pos: (B, 40, 3)
    region_score: (B*V, 40)
    v0: (B*V, 3)
    """
    B, K, _ = handle_pos.shape
    disp = transformation[:, :, :3]
    rot = transformation[:, :, 3:]
    rot = repr6d2mat_tensor(rot)
    # rot = quaternion_to_rotation_matrix(rot.view(B*K, 4).contiguous(), order=wxyz).view(B, K, 3, 3).contiguous()
    hd_disp = torch.repeat_interleave(disp, torch.bincount(batch), dim=0)  # (B*V, 40, 3)
    hd_rot = torch.repeat_interleave(rot, torch.bincount(batch), dim=0)  # (B*V, 40, 3, 3)
    hd_pos = torch.repeat_interleave(handle_pos, torch.bincount(batch), dim=0)  # (B*V, 40, 3)
    per_hd_v = torch.einsum("abcd,abd->abc", hd_rot, (v0[:, None] - hd_pos)) + hd_pos + hd_disp  # (B*V, 40, 3)
    v = torch.sum(region_score[:, :, None] * per_hd_v, 1)  # (B*V, 3)
    return v

def trans2handle(transformation, handle_pos):
    B, K, _ = handle_pos.shape   # bs 40 3
    disp = transformation[:, :, :3]         # B K 3
    return handle_pos + disp

def apply_root_rot(root_rotation, hm, batch, v):
    """
    use per-part trans+rot to reconstruct the mesh
    root_rotation: (B*V, 3)
    hm:  (B*V, K) score is to make the weight summation of each part as 1.   bs*V K, for each k in K, sum(V)=1
    v: (B*V, 3)
    """
    root_position = hm[:, 0] * v
    root_position = scatter_sum(root_position, batch, dim=0)        # B 3
    root_position_rep = torch.repeat_interleave(root_position, torch.bincount(batch), dim=0)   # B*V 3

    v_center = v - root_position_rep
    

def torch_geometric_fps(x, batch, K):
    with torch.no_grad():
        N = torch.max(batch)+1 # batch size
        N = N.cpu().item()
        batch_pts = []
        batch_idx = []
        for i in range(N):
            v = x[batch == i]
            pts, idx = fps((v, K, None))
            batch_pts.append(pts)
            batch_idx.append(idx)
    batch_pts = torch.stack(batch_pts, 0)
    return batch_pts


def gt_trans_mesh(pred_disp, hm, region_score, batch, v0, v1):
    with torch.no_grad():
        hd0 = scatter_sum(hm[:, :, None] * v0[:, None], batch, dim=0)
        hd1 = scatter_sum(hm[:, :, None] * v1[:, None], batch, dim=0)
        gt_trans = hd1 - hd0  # (B, K, 3)
    pred_rot = pred_disp[:, :, 3:]  # (B, K, 4)
    return handle2mesh(torch.cat((gt_trans, pred_rot), 2), hd0, region_score, batch, v0)


def arap_smooth(trans, center, sw, batch, v0, tpl_edge_index, arap_weight=-1):
    if arap_weight <= 0:
        return trans
    torch.set_grad_enabled(True)
    pred_v0 = handle2mesh(trans, center, sw, batch, v0)

    rot = trans[:,:,3:].clone().detach().requires_grad_(True)
    transl = trans[:,:,:3].clone().detach().requires_grad_(True)
    # pred_trans = trans.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([transl], lr=1e-2)
    l1 = torch.nn.L1Loss()

    max_iter = 300                              # 300
    min_i, min_loss = -1, 1000000
    for i in range(max_iter):
        optimizer.zero_grad()
        # pred_v = handle2mesh(pred_trans, center, sw, batch, v0)
        pred_v = handle2mesh(torch.cat([transl, rot],dim=-1), center, sw, batch, v0)
        loss_data = l1(pred_v, pred_v0)
        loss_arap = arap_loss(tpl_edge_index, pred_v, v0) * 100
        loss = loss_data + loss_arap * arap_weight
        loss.backward(retain_graph=True)
        loss_ = loss.detach().cpu().item()
        # print(f"iter {i}, data loss {loss_data.detach().cpu().item():.6f}"
        #       f", arap loss {loss_arap.detach().cpu().item():.6f}")
        optimizer.step()
        if loss_ < min_loss:
            min_loss = loss_
            min_i = i
        if i - min_i >= 20:
            break
    torch.set_grad_enabled(False)
    return torch.cat([transl, rot],dim=-1)


def arap_smooth2(trans, center, sw, batch, v0, tpl_edge_index, arap_weight=-1):
    if arap_weight <= 0:
        return trans
    torch.set_grad_enabled(True)
    
    rot = trans[:,:,3:].clone().detach().requires_grad_(True)
    transl = trans[:,:,:3].clone().detach().requires_grad_(True)
    transl_ori = transl.clone()
    optimizer = torch.optim.Adam([transl], lr=1e-2)
    l1 = torch.nn.L1Loss()

    max_iter = 300                              # 300
    min_i, min_loss = -1, 1000000
    for i in range(max_iter):
        optimizer.zero_grad()
        pred_v = handle2mesh(torch.cat([transl, rot],dim=-1), center, sw, batch, v0)
        loss_data = l1(transl_ori, transl)
        loss_arap = arap_loss(tpl_edge_index, pred_v, v0) * 1000
        loss = loss_data + loss_arap * arap_weight
        loss.backward(retain_graph=True)
        loss_ = loss.detach().cpu().item()
        optimizer.step()
        if loss_ < min_loss:
            min_loss = loss_
            min_i = i
        if i - min_i >= 10:
            break
    torch.set_grad_enabled(False)
    transl_ori[:,[13,2,11,8],:] = transl[:,[13,2,11,8],:]
    return torch.cat([transl_ori, rot],dim=-1)

def edge_loss(tpl_edge_index, pred_v, v0, loss_fn):
    # pred_v, v0: V 3
    edge_idx0 = tpl_edge_index[0,:]
    edge_idx1 = tpl_edge_index[1,:]

    t_edge_v0 = v0[edge_idx0, :]
    t_edge_v1 = v0[edge_idx1, :]

    p_edge_v0 = pred_v[edge_idx0, :]
    p_edge_v1 = pred_v[edge_idx1, :]

    dt = torch.sum((t_edge_v1 - t_edge_v0)**2, dim=-1)
    dp = torch.sum((p_edge_v0 - p_edge_v1)**2, dim=-1)

    return loss_fn(dt, dp)



def edge_smooth(trans, center, sw, batch, v0, tpl_edge_index, edge_weight=-1):
    if edge_weight <= 0:
        return trans
    
    torch.set_grad_enabled(True)
    pred_v0 = handle2mesh(trans, center, sw, batch, v0)

    rot = trans[:,:,3:].clone().detach().requires_grad_(True)
    transl = trans[:,:,:3].clone().detach().requires_grad_(True)
    # pred_trans = trans.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([transl], lr=1e-2)
    l1 = torch.nn.L1Loss()
    l2 = torch.nn.MSELoss()

    max_iter = 100                              # 300
    min_i, min_loss = -1, 1000000
    for i in range(max_iter):
        optimizer.zero_grad()
        # pred_v = handle2mesh(pred_trans, center, sw, batch, v0)
        pred_v = handle2mesh(torch.cat([transl, rot],dim=-1), center, sw, batch, v0)
        loss_data = l1(pred_v, pred_v0)
        loss_edge = edge_loss(tpl_edge_index, pred_v, v0, l2) * 1e5
        loss = loss_data  + loss_edge * edge_weight
        print(loss_data, loss_edge)
        loss.backward(retain_graph=True)
        loss_ = loss.detach().cpu().item()
        # print(f"iter {i}, data loss {loss_data.detach().cpu().item():.6f}"
        #       f", arap loss {loss_arap.detach().cpu().item():.6f}")
        optimizer.step()
        if loss_ < min_loss:
            min_loss = loss_
            min_i = i
        if i - min_i >= 10:
            break
    torch.set_grad_enabled(False)
    return torch.cat([transl, rot],dim=-1)

def handle_dis_loss(pred_hd, t_handle, loss_fn):
    handle_map = [[2,11],[11,26],[26,27], [13,8],[8,19],[19,18], [7,23],[23,29], [1,3],[3,25]]
    handle_0 = [handle_map[i][0] for i in range(len(handle_map))]
    handle_1 = [handle_map[i][1] for i in range(len(handle_map))]

    pred_pos_0 = pred_hd[:, handle_0, :]
    pred_pos_1 = pred_hd[:, handle_1, :]
    t_pos_0 = t_handle[:, handle_0, :]
    t_pos_1 = t_handle[:, handle_1, :]
    handle_dis = torch.sum((pred_pos_1 - pred_pos_0)**2, dim=-1)
    t_dis = torch.sum((t_pos_1 - t_pos_0)**2, dim=-1)

    return loss_fn(handle_dis, t_dis)




def arap_handle_smooth(trans, center, sw, batch, v0, tpl_edge_index, arap_weight=-1, handle_weight=-1):
    if arap_weight <= 0 and handle_weight<=0:
        return trans
    if arap_weight<0: arap_weight=0.0
    if handle_weight<0: handle_weight=0.0

    torch.set_grad_enabled(True)
    pred_v0 = handle2mesh(trans, center, sw, batch, v0)
    pred_hd = trans2handle(trans, center)

    rot = trans[:,:,3:].clone().detach().requires_grad_(True)
    transl = trans[:,:,:3].clone().detach().requires_grad_(True)
    # pred_trans = trans.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([transl], lr=1e-2)
    l1 = torch.nn.L1Loss()

    max_iter = 300                              # 300
    min_i, min_loss = -1, 1000000
    for i in range(max_iter):
        optimizer.zero_grad()
        # pred_v = handle2mesh(pred_trans, center, sw, batch, v0)
        pred_v = handle2mesh(torch.cat([transl, rot],dim=-1), center, sw, batch, v0)
        loss_data = l1(pred_v, pred_v0)
        loss_arap = arap_loss(tpl_edge_index, pred_v, v0) * 100
        loss_handle = handle_dis_loss(pred_hd, center, l1)
        loss = 0*loss_data + loss_arap * 0*arap_weight + handle_weight *loss_handle
        print(loss_data, loss_arap, loss_handle)
        loss.backward(retain_graph=True)
        loss_ = loss.detach().cpu().item()
        # print(f"iter {i}, data loss {loss_data.detach().cpu().item():.6f}"
        #       f", arap loss {loss_arap.detach().cpu().item():.6f}")
        optimizer.step()
        if loss_ < min_loss:
            min_loss = loss_
            min_i = i
        if i - min_i >= 100:
            break
    torch.set_grad_enabled(False)
    return torch.cat([transl, rot],dim=-1)


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=0.15, beta=0.01,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
