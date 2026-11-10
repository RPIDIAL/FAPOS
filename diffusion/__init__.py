# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
#from . import gaussian_diffusion_uncond as gdu
#from . import gaussian_diffusion_cf as gdcf
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )

#def create_diffusion_uncond(
#    timestep_respacing,
#    noise_schedule="linear", 
#    use_kl=False,
#    sigma_small=False,
#    predict_xstart=False,
#    learn_sigma=True,
#    rescale_learned_sigmas=False,
#    diffusion_steps=1000
#):
#    betas = gdu.get_named_beta_schedule(noise_schedule, diffusion_steps)
#    if use_kl:
#        loss_type = gdu.LossType.RESCALED_KL
#    elif rescale_learned_sigmas:
#        loss_type = gdu.LossType.RESCALED_MSE
#    else:
#        loss_type = gdu.LossType.MSE
#    if timestep_respacing is None or timestep_respacing == "":
#        timestep_respacing = [diffusion_steps]
#    return SpacedDiffusion(
#        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
#        betas=betas,
#        model_mean_type=(
#            gdu.ModelMeanType.EPSILON if not predict_xstart else gdu.ModelMeanType.START_X
#        ),
#        model_var_type=(
#            (
#                gdu.ModelVarType.FIXED_LARGE
#                if not sigma_small
#                else gdu.ModelVarType.FIXED_SMALL
#            )
#            if not learn_sigma
#            else gdu.ModelVarType.LEARNED_RANGE
#        ),
#        loss_type=loss_type
#        # rescale_timesteps=rescale_timesteps,
#    )
#
#def create_diffusion_cf(
#    timestep_respacing,
#    noise_schedule="linear", 
#    use_kl=False,
#    sigma_small=False,
#    predict_xstart=False,
#    learn_sigma=True,
#    rescale_learned_sigmas=False,
#    diffusion_steps=1000
#):
#    betas = gdcf.get_named_beta_schedule(noise_schedule, diffusion_steps)
#    if use_kl:
#        loss_type = gdcf.LossType.RESCALED_KL
#    elif rescale_learned_sigmas:
#        loss_type = gdcf.LossType.RESCALED_MSE
#    else:
#        loss_type = gdcf.LossType.MSE
#    if timestep_respacing is None or timestep_respacing == "":
#        timestep_respacing = [diffusion_steps]
#    return SpacedDiffusion(
#        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
#        betas=betas,
#        model_mean_type=(
#            gdcf.ModelMeanType.EPSILON if not predict_xstart else gdcf.ModelMeanType.START_X
#        ),
#        model_var_type=(
#            (
#                gdcf.ModelVarType.FIXED_LARGE
#                if not sigma_small
#                else gdcf.ModelVarType.FIXED_SMALL
#            )
#            if not learn_sigma
#            else gdcf.ModelVarType.LEARNED_RANGE
#        ),
#        loss_type=loss_type
#        # rescale_timesteps=rescale_timesteps,
#    )
#