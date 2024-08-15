import torch
import torch.nn.functional as F

def compute_ssim(img1, img2, window_size=11, sigma=1.5, size_average=True):
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    The SSIM index is used to measure the similarity between two images. It considers changes in structural information, luminance, and contrast.

    Parameters:
    - img1 (torch.Tensor): The first input image tensor of shape (N, C, H, W) where N is batch size, C is number of channels, H is height, and W is width.
    - img2 (torch.Tensor): The second input image tensor of the same shape as img1.
    - window_size (int, optional): The size of the Gaussian window used in SSIM calculation. Default is 11.
    - sigma (float, optional): The standard deviation of the Gaussian window. Default is 1.5.
    - size_average (bool, optional): If True, the SSIM index is averaged over all the images in the batch. If False, SSIM is calculated per image and then averaged. Default is True.

    Returns:
    - torch.Tensor: The SSIM index value. If `size_average` is True, returns a scalar tensor; otherwise, returns a tensor with shape (N,).

    Example:
    >>> image1_tensor = torch.rand((1, 1, 256, 256))  # (batch_size, channel, height, width)
    >>> image2_tensor = torch.rand((1, 1, 256, 256))
    >>> ssim_value = compute_ssim(image1_tensor, image2_tensor)
    >>> print(ssim_value.item())
    """
    def gaussian(window_size, sigma):
        gauss = torch.tensor([-(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)])
        gauss = torch.exp(gauss)
        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    channel = img1.size(1)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

# 테스트를 위한 예시 이미지 텐서 생성
# image1_tensor = torch.rand((3, 1, 256, 256))  # (batch_size, channel, height, width)
# image2_tensor = torch.rand((3, 1, 256, 256))

# SSIM 계산
#ssim_value = compute_ssim(image1_tensor, image2_tensor)
#print(f'SSIM: {ssim_value.item()}')

