import numpy as np
import torch
import torch.nn as nn

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
    num_batches = len(dataloader)
    dataset_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    print(f"Dataset size: {dataset_size}")
    print(f"Number of batches: {num_batches}")
    print(f"Batch size: {batch_size}")
    
    # 첫 번째 배치의 정보를 출력
    for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}/{num_batches}")
        print(f"  Data shape: {batch_data.shape}")
        print(f"  Labels shape: {batch_labels.shape}")
        print(f"  Data type: {batch_data.dtype}")
        print(f"  Labels type: {batch_labels.dtype}")
        
        # 통계 정보 계산 및 출력
        batch_data_mean = batch_data.mean().item()
        batch_data_std = batch_data.std().item()
        print(f"  Data mean: {batch_data_mean:.2f}")
        print(f"  Data std: {batch_data_std:.2f}")
        
        batch_labels_mean = batch_labels.mean().item()
        batch_labels_std = batch_labels.std().item()
        print(f"  Label mean: {batch_labels_mean:.2f}")
        print(f"  Label std: {batch_labels_std:.2f}")
        
        # 배치 하나만 요약하기 위해 break
        break
        
def smoothObject(obj, iter):
    obj_tensor = torch.Tensor(obj).unsqueeze(0)
    for i in range(iter):
        # print(obj_tensor.shape)
        obj_tensor = nn.functional.avg_pool2d(obj_tensor, 3, 1, 1)
        # print(obj_tensor.shape)
        
    return obj_tensor[0].numpy()