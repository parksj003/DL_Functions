def restore_image(batch_image, mean, std):
        return batch_image* std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1) 
