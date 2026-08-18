# Environment record

Verified on 18 August 2026.

| Component | Version / result |
| --- | --- |
| Windows | 10.0.26200.9168 |
| WSL | 2.6.1.0 |
| WSL kernel | 6.6.87.2-microsoft-standard-WSL2 |
| Distribution | Ubuntu 24.04.4 LTS |
| Python | 3.12.3 |
| Windows NVIDIA driver | 592.82 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU memory | 8,188 MiB reported by `nvidia-smi` |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| CUDA allocation | Passed |
| CUDA matrix multiplication | Passed |
| CUDA autograd/backward | Passed |
| ManiSkill | 3.0.1 |
| SAPIEN | 3.0.3 |
| PushCube CPU-state | Passed; deterministic record/replay |
| PushCube WSL RGB-D fallback | Passed; 128x128 RGB and depth at about 39 steps/s |
| PushCube WSL GPU RGB-D | Unsupported; NVIDIA Vulkan ICD unavailable in WSL |

The default WSL distribution is `docker-desktop`, so project commands must explicitly use
`wsl -d Ubuntu-24.04` or the Windows default should be changed intentionally later.

ManiSkill's published support table marks WSL GPU simulation and rendering unsupported.
The project therefore tests CPU-state and GPU-RGB-D modes independently and records the
actual local result rather than assuming support.
