import numpy as np
import torch
import torch.nn as nn
import LightPipes as LP
import gc
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def montage(images, grid_shape=None):
    """
    여러 이미지를 하나의 큰 이미지로 합치는 몽타주(montage)를 생성하는 함수.

    Args:
        images (numpy.ndarray): (N, H, W) 크기의 이미지 배열.
        grid_shape (tuple, optional): (rows, cols)의 형태로 출력 몽타주 그리드의 크기.
                                      제공되지 않으면 정사각형에 가까운 형태로 자동 결정.

    Returns:
        numpy.ndarray: 합쳐진 몽타주 이미지.
    """
    num_images = images.shape[0]
    H, W = images.shape[1:]

    if grid_shape is None:
        grid_cols = int(np.ceil(np.sqrt(num_images)))
        grid_rows = int(np.ceil(num_images / grid_cols))
    else:
        grid_rows, grid_cols = grid_shape

    # 검증
    if grid_rows * grid_cols < num_images:
        raise ValueError("그리드의 크기가 이미지 수보다 작습니다.")

    # 몽타주 캔버스 생성
    montage_image = np.zeros((grid_rows * H, grid_cols * W), dtype=images.dtype)

    for idx in range(num_images):
        row = idx // grid_cols
        col = idx % grid_cols
        montage_image[row*H:(row+1)*H, col*W:(col+1)*W] = images[idx]

    return montage_image

def print_dataloader_summary(dataloader):
    
    def get_channel_stats(data):
        channel_means = []
        channel_stds = []
        channel_mins = []
        channel_maxs = []
        
        for c in range(data.size(1)):
            channel_data = data[:, c, :, :]
            channel_means.append(round(channel_data.mean().item(), 2))
            channel_stds.append(round(channel_data.std().item(), 2))
            channel_mins.append(round(channel_data.min().item(), 2))
            channel_maxs.append(round(channel_data.max().item(), 2))
        
        return channel_means, channel_stds, channel_mins, channel_maxs

    def print_channel_stats(channel_means, channel_stds, channel_mins, channel_maxs, name=""):
        for c in range(len(channel_means)):
            print(f"    {name} Channel {c}: mean = {channel_means[c]:.2f}, std = {channel_stds[c]:.2f}, "
                  f"min = {channel_mins[c]:.2f}, max = {channel_maxs[c]:.2f}")
        print()  # 줄바꿈
        
    num_batches = len(dataloader)
    dataset_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    
    print(f"Dataset size: {dataset_size}")
    print(f"Number of batches: {num_batches}")
    print(f"Batch size: {batch_size}\n")
    
    # 첫 번째 배치의 정보를 출력하고 해당 배치 데이터를 반환
    for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}/{num_batches} Summary")
        
        # 데이터 정보 출력
        print("  [Data]")
        print(f"    Data shape: {batch_data.shape}")
        print(f"    Data type: {batch_data.dtype}")
        
        if batch_data.ndim == 4:  # assuming [B, C, H, W] format
            channel_means, channel_stds, channel_mins, channel_maxs = get_channel_stats(batch_data)
            overall_mean = batch_data.mean().item()
            overall_std = batch_data.std().item()
            overall_min = batch_data.min().item()
            overall_max = batch_data.max().item()
            print(f"    Data mean: {overall_mean:.2f} (Channels: {channel_means})")
            print(f"    Data std: {overall_std:.2f} (Channels: {channel_stds})")
            print(f"    Data min: {overall_min:.2f} (Channels: {channel_mins})")
            print(f"    Data max: {overall_max:.2f} (Channels: {channel_maxs})\n")
            
            # 채널별 통계 출력
            # print_channel_stats(channel_means, channel_stds, channel_mins, channel_maxs, "Data")
        
        # 라벨 정보 출력
        print("  [Labels]")
        print(f"    Labels shape: {batch_labels.shape}")
        print(f"    Labels type: {batch_labels.dtype}")
        
        if batch_labels.dtype == torch.float32:
            overall_mean = batch_labels.mean().item()
            overall_std = batch_labels.std().item()
            print(f"    Label mean: {overall_mean:.2f}")
            print(f"    Label std: {overall_std:.2f}")
        
        overall_min = batch_labels.min().item()
        overall_max = batch_labels.max().item()
        print(f"    Label min: {overall_min:.2f}")
        print(f"    Label max: {overall_max:.2f}\n")
        
        if batch_labels.ndim == 4:  # assuming [B, C, H, W] format
            channel_means, channel_stds, channel_mins, channel_maxs = get_channel_stats(batch_labels)
            print_channel_stats(channel_means, channel_stds, channel_mins, channel_maxs, "Labels")
        
        # 첫 번째 배치를 반환
        return batch_data, batch_labels

# batch_example = print_dataloader_summary(dataloader_train)
        
def smoothObject(obj, iter):
    obj_tensor = torch.Tensor(obj).unsqueeze(0)
    for i in range(iter):
        # print(obj_tensor.shape)
        obj_tensor = nn.functional.avg_pool2d(obj_tensor, 3, 1, 1)
        # print(obj_tensor.shape)
        
    return obj_tensor[0].numpy()
    
def GPU_summary():
    import torch

    print("\n===================================================")
    if torch.cuda.is_available():

        print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        gpu_id = 0  # 확인하려는 GPU의 ID (기본적으로 0번 GPU)    
        # GPU의 총 메모리 용량 (바이트 단위)
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory

        # 할당된 메모리 (바이트 단위)
        allocated_memory = torch.cuda.memory_allocated(gpu_id)

        # 캐시된 메모리 (바이트 단위)
        cached_memory = torch.cuda.memory_reserved(gpu_id)

        print(f"GPU {gpu_id} 메모리 정보:")
        print(f"  - 전체 메모리: {total_memory / 1024**2:.2f} MB")
        print(f"  - 할당된 메모리: {allocated_memory / 1024**2:.2f} MB")
        print(f"  - 캐시된 메모리: {cached_memory / 1024**2:.2f} MB")
    else:
        print("CUDA GPU를 사용할 수 없습니다.")
    # GPU 메모리 정리
    
def LP_layer_batch(phase_batch, wavelength, gaussian_width, z, size, N):
# 예시 데이터 (배치 크기 8, 채널 1, 256x256 이미지)
# wavelength=632.8*LP.nm; #wavelength of the HeNe laser used
# gaussian_width = 2*LP.mm
# z = 2*LP.m 
# size=20*LP.mm; #The CCD-sensor has an area of size x size (NB LightPipes needs square grids!)
# N = 256 #CCD pixels
    F_init = LP.Begin(size, wavelength, N)
    F_init = LP.GaussAperture(gaussian_width, 0, 0, 1, F_init)
    processed_images = []
    for phase in phase_batch:
        # 각 배치에 대해 LP 연산 수행
        phase = phase[0].detach().cpu()
        F_out = LP.SubPhase(F_init, phase)
        F_out = LP.Lens(z, F_out)
        F_out = LP.Forvard(z, F_out)  # Propagate to the far field
        I = LP.Intensity(F_out)
        processed_image = torch.from_numpy(I).to(torch.float32).to(DEVICE)
        processed_images.append(processed_image)
    
    # 처리된 이미지를 다시 하나의 텐서로 합침
    return torch.stack(processed_images).unsqueeze(1)
#beam_recon = LP_layer_batch(recon,wavelength, gaussian_width, z, size, N)

def restore_image(batch_image, mean, std):
        return batch_image* std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1) 


def delete_model(gabage_model):
    print("Model is removed from GPU!")
    del gabage_model
    gc.collect() 
    torch.cuda.empty_cache()

