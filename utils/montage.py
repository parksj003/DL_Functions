import numpy as np

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
