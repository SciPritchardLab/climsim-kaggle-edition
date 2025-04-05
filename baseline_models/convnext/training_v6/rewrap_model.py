import torch
import modulus

from wrap_model import WrappedModel

device = torch.device('cuda')

mdlus_paths = []

def load_ckpt(ckpt_path):
    model = modulus.Module.from_checkpoint(ckpt_path).to(device)
    wrapped_model = WrappedModel(
        original_model=model,
        input_sub=model.input_sub.cpu().numpy(),
        input_div=model.input_div.cpu().numpy(),
        out_scale=model.out_scale.cpu().numpy(),
        qn_lbd=model.qn_lbd.cpu().numpy()
    )
    save_path_wrapped = os.path.join(wrapped_directory, filename.replace('.mdlus', '_wrapped.pt'))
    scripted_model_wrapped = torch.jit.script(wrapped_model)
    scripted_model_wrapped = scripted_model_wrapped.eval()
    scripted_model_wrapped.save(save_path_wrapped)

for mdlus_path in mdlus_paths:
    load_ckpt(mdlus_path)