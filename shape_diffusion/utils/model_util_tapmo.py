from model.mdm import MDM
from model.mdm_t6d import MDM_T6d
from diffusion import gaussian_diffusion as gd
from diffusion import gs_diffusion_t6d_dis as gd_6d_disc
from diffusion.respace import SpacedDiffusion, SpacedDiffusion_t6d_disc, space_timesteps
from model.motion_discriminator import MotionDiscriminator
# from config import SMPLH_PATH, PART_PATH


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    # assert all([k.startswith('clip_model.') for k in missing_keys])


def load_model_wo_hdclip(model, state_dict):
    missing_keys, unexphdected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexphdected_keys) == 0

def create_model_and_diffusion(args, data):
    if args.dataset == 'humanml3d':
        model = MDM(**get_model_args(args, data))
    elif args.dataset == 't6d_mixamo' or args.dataset == 't6d_mixrig':
        model = MDM_T6d(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args, data):

    # default args
    # clip_version = 'ViT-B/32'
    clip_version = './deps/clip-vit-large-patch14'
    SMPLH_PATH = "../handle_predictor/smplh"
    PART_PATH = "../handle_predictor/smplh"

    action_emb = 'tensor'
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset in ['kit', 'humanml', "t6d_mixrig"]:
        cond_mode = 'text'
    else:
        cond_mode = 'action'
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    data_rep = '6d'
    njoints = 30
    nfeats = 9

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'num_actions': num_actions,
            'translation': True, "smpl_path": SMPLH_PATH , "part_path": PART_PATH, 
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset}


def create_gaussian_diffusion(args):
    # default params

    if args.dataset == 'humanml3d':
        predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
        steps = 1000
        scale_beta = 1.  # no scaling
        timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
        learn_sigma = False
        rescale_timesteps = False

        betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
        loss_type = gd.LossType.MSE

        if not timestep_respacing:
            timestep_respacing = [steps]

        return SpacedDiffusion(
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not args.sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
            lambda_mesh=args.lambda_mesh,
            lambda_arap=args.lambda_arap,
            lambda_handle=args.lambda_handle,
        )
    else:
        predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
        steps = 1000
        scale_beta = 1.  # no scaling
        timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
        learn_sigma = False
        rescale_timesteps = False

        betas = gd_6d_disc.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
        loss_type = gd_6d_disc.LossType.MSE

        if not timestep_respacing:
            timestep_respacing = [steps]

        return SpacedDiffusion_t6d_disc(
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd_6d_disc.ModelMeanType.EPSILON if not predict_xstart else gd_6d_disc.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd_6d_disc.ModelVarType.FIXED_LARGE
                    if not args.sigma_small
                    else gd_6d_disc.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd_6d_disc.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
            lambda_gen=args.lambda_gen,
        )




def creat_discriminator():
    discriminator = MotionDiscriminator(feature_pool="attention")
    return discriminator