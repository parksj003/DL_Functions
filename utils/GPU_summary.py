import torch 

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
