import torch

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
