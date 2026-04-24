import torch
import torch.nn as nn
import torch.optim as optim
import time


def run_gpu_test():
    # 1. Check if CUDA (GPU support) is available
    print("--- System Check ---")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if not cuda_available:
        print("RESULT: No GPU found. The script will run on your CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Device being used: {device}")

    # 2. Define a simple Neural Network
    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    ).to(device)  # <--- Move model to GPU

    # 3. Create dummy data (Large enough to stress the GPU slightly)
    # 128 images, 2048 features each
    inputs = torch.randn(128, 2048).to(device)
    targets = torch.randn(128, 10).to(device)

    # 4. Define Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training Loop
    print("\n--- Starting Training Test ---")
    start_time = time.time()

    for epoch in range(500):
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/500], Loss: {loss.item():.4f}")

    end_time = time.time()

    print("\n--- Result ---")
    print(f"Total time for 500 epochs: {end_time - start_time:.4f} seconds")
    if cuda_available:
        print("Success! Your GPU was used for this training.")
    else:
        print("Test complete, but it ran on the CPU.")


if __name__ == "__main__":
    run_gpu_test()
