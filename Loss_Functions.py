import torch.nn.functional as nnF  # nnF로 변경
import torch.nn as nn
# # 모델 생성
# ssim_loss = SSIMLoss()

# # 테스트 이미지 생성
# img1 = torch.randn(1, 1, 256, 256)  # N, C, H, W
# img2 = torch.randn(1, 1, 256, 256)  # N, C, H, W

# # SSIM 손실 계산
# loss = ssim_loss(img1, img2)
# print(f"SSIM Loss: {loss.item()}")


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window(window_size).repeat(channel, 1, 1, 1).to(dtype=torch.float32)

    def create_window(self, window_size, sigma=1.5):
        gauss = self.gaussian(window_size, sigma)
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        return _2D_window

    def gaussian(self, window_size, sigma):
        gauss = torch.exp(-torch.arange(window_size).float() ** 2 / (2 * sigma ** 2))
        return gauss

    def forward(self, img1, img2):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        window = self.window.to(img1.device)
        mu1 = nnF.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = nnF.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = nnF.conv2d(img1 ** 2, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = nnF.conv2d(img2 ** 2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = nnF.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

