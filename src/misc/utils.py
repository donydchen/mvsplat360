import torch
import random


def split_into_chunks(lst_len, chunk_size, overlap):
    chunks = []
    start = 0
    while start < lst_len:
        end = start + chunk_size
        chunks.append([start, end])
        start += chunk_size - overlap
        if end >= lst_len:
            break
    return chunks


def gauss_noise_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)

    sigma = 0.1
    sigma = [random.choice([0.0, 0.0, sigma]) for _ in range(img.shape[0])]
    sigma = torch.tensor(sigma).to(img).reshape(img.shape[0], *([1] * (img.dim() - 1)))
    out = img + sigma * torch.randn_like(img)
    out = torch.clip(out, 0.0, 1.0)

    if out.dtype != dtype:
        out = out.to(dtype)

    return out


def random_regular_mask(img):
    """
    https://github.com/lyndonzheng/TFill/blob/main/util/task.py
    Generate a random regular mask
    :param img: original image size  C*H*W
    :return: mask
    """
    mask = torch.ones_like(img)[0:1, :, :]
    s = img.size()
    N_mask = random.randint(1, 5)
    lim_x = s[1] - s[1] / (N_mask + 1)
    lim_y = s[2] - s[2] / (N_mask + 1)
    for _ in range(N_mask):
        x = random.randint(0, int(lim_x))
        y = random.randint(0, int(lim_y))
        range_x = x + random.randint(
            int(s[1] / (N_mask + 7)), min(int(s[1] - x), int(s[1] / 2))
        )
        range_y = y + random.randint(
            int(s[2] / (N_mask + 7)), min(int(s[2] - y), int(s[2] / 2))
        )
        mask[:, int(x) : int(range_x), int(y) : int(range_y)] = 0
    return mask


def random_erase_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)

    mask_prob = 0.33
    masks = [
        (
            random_regular_mask(img[0]).to(img[0])
            if random.random() < mask_prob
            else torch.ones_like(img[0, 0:1]).to(img[0])
        )
        for _ in range(img.shape[0])
    ]
    masks = torch.stack(masks)
    out = img * masks
    out = torch.clip(out, 0.0, 1.0)

    if out.dtype != dtype:
        out = out.to(dtype)

    return out
