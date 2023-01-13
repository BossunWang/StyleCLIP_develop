
from argparse import Namespace
import time
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import clip
from contextlib import contextmanager
from tqdm.auto import tqdm


from encoder4editing.utils.common import tensor2im
from encoder4editing.models.psp import pSp
from global_torch.manipulate import Manipulator
from global_torch.StyleCLIP import GetDt, GetBoundary
import dlib
from encoder4editing.utils.alignment import align_face


def run_alignment(image_path, predictor):
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def display_alongside_source_image(result_image, source_image, resize_dims):
    res = np.concatenate([np.array(source_image.resize(resize_dims)),
                          np.array(result_image.resize(resize_dims))], axis=1)
    return Image.fromarray(res)


def run_on_batch(inputs, net, experiment_type):
    images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    if experiment_type == 'cars_encode':
        images = images[:, :, 32:224, :]
    return images, latents


# Mute GetBoundary()
# https://stackoverflow.com/a/25061573
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def gen_image(fs3, dt, M, dlatent_tmp, beta, alpha, i):
    M.alpha = [alpha]
    with suppress_stdout():
        boundary_tmp2, c = GetBoundary(fs3, dt, M,threshold=beta)
    codes = M.MSCode(dlatent_tmp, boundary_tmp2)
    out = M.GenerateImg(codes)
    Image.fromarray(out[0,0]).save(f"manipulated_results/{i:04d}.png")


def GetBoundaryNum(fs3, dt, threshold):
    tmp = np.dot(fs3, dt)
    select = np.abs(tmp) < threshold
    num_c = np.sum(~select)
    return num_c


def main():
    os.makedirs("latents", exist_ok=True)
    os.makedirs("manipulated_results", exist_ok=True)

    # input dataset name
    dataset_name = 'ffhq'  # @param ['ffhq'] {allow-input: true}

    if not os.path.isfile('global_torch/model/' + dataset_name + '.pkl'):
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/'
        name = 'stylegan2-' + dataset_name + '-config-f.pkl'
        os.system('wget ' + url + name + '  -P  global_torch/model/')
        os.system('mv global_torch/model/' + name + ' global_torch/model/' + dataset_name + '.pkl')

    # input prepare data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    network_pkl = 'global_torch/model/' + dataset_name + '.pkl'
    device = torch.device('cuda')
    M = Manipulator()
    M.device = device
    G = M.LoadModel(network_pkl, device)
    M.G = G
    M.SetGParameters()
    num_img = 100_000
    M.GenerateS(num_img=num_img)
    M.GetCodeMS()
    np.set_printoptions(suppress=True)

    file_path = 'global_torch/npy/' + dataset_name + '/'
    fs3 = np.load(file_path + 'fs3.npy')

    # @title e4e setup
    # @ e4e setup
    experiment_type = 'ffhq_encode'
    EXPERIMENT_ARGS = {"model_path": "pretrained/e4e_ffhq_encode.pt", 'transform': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    resize_dims = (256, 256)

    model_path = EXPERIMENT_ARGS['model_path']
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    # pprint.pprint(opts)  # Display full options used
    # update the training options
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')

    # @title Align image
    image_path = "/media/user/HGST/Bossun/face_dataset/Emotion/RAF-DB/basic-20201119T055425Z-001/basic/Image/original/train_09860.jpg"  # @param {type: "string"}
    original_image = Image.open(image_path)
    original_image = original_image.convert("RGB")
    predictor = dlib.shape_predictor("pretrained/shape_predictor_68_face_landmarks.dat")

    if experiment_type == "ffhq_encode":
        input_image = run_alignment(image_path, predictor)
    else:
        input_image = original_image

    input_image.resize(resize_dims)
    plt.imshow(input_image)
    plt.savefig("aligned.png")

    # @title Invert the image
    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(input_image)

    with torch.no_grad():
        images, latents = run_on_batch(transformed_image.unsqueeze(0), net, experiment_type)
        result_image, latent = images[0], latents[0]
    latent_path = 'latents/{}_latents.pt'.format(image_path.split("/")[-1].replace(".jpg", ""))
    torch.save(latents, latent_path)

    # Display inversion:
    pixel2style2pixel_img = display_alongside_source_image(tensor2im(result_image), input_image, resize_dims)
    plt.imshow(pixel2style2pixel_img)
    plt.savefig("pixel2style2pixel_img.png")

    # Choose Image Index: relevant only when editing generated image
    img_index = 1  # @param {type:"number"}

    # Choose Mode (and show input image)
    img_index = 0
    latents = torch.load(latent_path)
    dlatents_loaded = M.G.synthesis.W2S(latents)

    img_indexs = [img_index]
    dlatents_loaded = M.S2List(dlatents_loaded)
    dlatent_tmp = [tmp[img_indexs] for tmp in dlatents_loaded]
    M.num_images = len(img_indexs)

    M.alpha = [0]
    M.manipulate_layers = [0]
    codes, out = M.EditOneC(0, dlatent_tmp)
    original = Image.fromarray(out[0, 0]).resize((512, 512))
    M.manipulate_layers = None
    plt.imshow(original)
    plt.savefig("original.png")

    # input text description
    neutral = 'face'  # @param {type:"string"}
    # target = 'smiling face'  # @param {type:"string"}
    target = 'a face that looks fearful'  # @param {type:"string"}
    classnames = [target, neutral]
    dt = GetDt(classnames, model)

    # # modify manipulation strength (alhpa) and disentangle threshold (beta)
    # # beta=0.1
    # # alpha=1
    # beta = 0.15  # @param {type:"slider", min:0.08, max:0.3, step:0.01}
    # alpha = 10.  # @param {type:"slider", min:-10, max:10, step:0.1}
    # # For color transformation, usually 10-20 channels is enough.
    # # For large structure change (for example, Hi-top fade), usually 100-200 channels are required.
    # M.alpha = [alpha]
    # boundary_tmp2, c = GetBoundary(fs3, dt, M, threshold=beta)
    # codes = M.MSCode(dlatent_tmp, boundary_tmp2)
    # out = M.GenerateImg(codes)
    # generated = Image.fromarray(out[0, 0])  # .resize((512,512))
    #
    # plt.figure(figsize=(20, 7), dpi=100)
    # plt.subplot(1, 2, 1)
    # plt.imshow(original)
    # plt.title('original')
    # plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(generated)
    # plt.title('manipulated')
    # plt.axis('off')
    # plt.savefig("manipulated.png")

    # video
    # Renders a video interpolating from the base image with provided beta to the target_alpha.
    # (target_alpha can be positive or negative)

    target_alpha = 6.  # @param {type:"number"}
    num_frames = 10  # @param {type:"number"}
    frame_rate = 60  # @param {type:"number"}

    # select beta adaptively
    betas = np.linspace(0.08, 0.3, 10)
    best_beta = 0.08
    best_num_c = 0
    for b in betas:
        num_c = GetBoundaryNum(fs3, dt, b)
        if 100 < num_c < 500:
            best_beta = b
            best_num_c = num_c

    print("best beta:", best_beta)
    print("best_num_c:", best_num_c)
    alphas = np.linspace(0, target_alpha, num_frames)
    beta = best_beta

    print("Generating Frames:")
    for i, alpha in tqdm(enumerate(alphas), total=num_frames):
        gen_image(fs3, dt, M, dlatent_tmp, beta, alpha, i)


if __name__ == '__main__':
    main()