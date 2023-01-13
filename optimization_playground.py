import torch
from optimization.run_optimization import main
from argparse import Namespace
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt


def run():
    experiment_type = 'edit'  # @param ['edit', 'free_generation']
    description = 'A person with smile'  # @param {type:"string"}
    latent_path = None  # @param {type:"string"}
    optimization_steps = 40  # @param {type:"number"}
    l2_lambda = 0.008  # @param {type:"number"}
    id_lambda = 0.005  # @param {type:"number"}
    stylespace = False  # @param {type:"boolean"}
    create_video = True  # @param {type:"boolean"}
    use_seed = True  # @param {type:"boolean"}
    seed = 41  # @param {type: "number"}

    # @title Additional Arguments
    args = {
        "description": description,
        "ckpt": "pretrained/stylegan2-ffhq-config-f.pt",
        "stylegan_size": 1024,
        "lr_rampup": 0.05,
        "lr": 0.1,
        "step": optimization_steps,
        "mode": experiment_type,
        "l2_lambda": l2_lambda,
        "id_lambda": id_lambda,
        'work_in_stylespace': stylespace,
        "latent_path": latent_path,
        "truncation": 0.7,
        "save_intermediate_image_every": 1 if create_video else 20,
        "results_dir": "results",
        "ir_se50_weights": "pretrained/model_ir_se50.pth"
    }

    if use_seed:
        torch.manual_seed(seed)
    result = main(Namespace(**args))

    # @title Visualize Result
    result_image = ToPILImage()(
        make_grid(result.detach().cpu(), normalize=True, scale_each=True, range=(-1, 1), padding=0))
    h, w = result_image.size
    result_image.resize((h // 2, w // 2))
    plt.imshow(result_image)
    plt.show()


if __name__ == '__main__':
    run()