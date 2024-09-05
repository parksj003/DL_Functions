import torch
import gc

def delete_model(gabage_model):
    print("Model is removed from GPU!")
    del gabage_model
    gc.collect() 
    torch.cuda.empty_cache()
