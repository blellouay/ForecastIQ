# Windows GPU diagnosis and user-controlled setup

The audited project environment is CPU-only: the interpreter is `.venv\Scripts\python.exe`, PyTorch is `2.15.0+cpu`, `torch.version.cuda` is `None`, CUDA availability is false, and `nvidia-smi` failed to initialize NVML.

Run these read-only diagnostics in PowerShell:

```powershell
where.exe python
.\.venv\Scripts\python.exe -m pip show torch
.\.venv\Scripts\python.exe -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
nvidia-smi
.\.venv\Scripts\python.exe -m jupyter kernelspec list
```

Repair or update the NVIDIA driver first if `nvidia-smi` still fails. Do not install a CUDA wheel until it identifies the GTX 1650.

Then use the official PyTorch **Start Locally** selector with Windows, Pip, Python, and a supported CUDA platform. CUDA 11.8 is a conservative wheel target for this GPU. Its project-environment command is:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This installation command was not executed. After installation, restart Jupyter completely, select `forecastiq-venv`, rerun diagnostics, and execute Notebook 3D in fast mode before research mode. If resolution conflicts with the pinned environment, create a separate GPU virtual environment instead of changing the working CPU environment.
