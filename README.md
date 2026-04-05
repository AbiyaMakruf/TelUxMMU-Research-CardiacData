# Install UV

## Windows
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## macOS and Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup Environment

From the project root, create and sync the environment from `pyproject.toml`:

```bash
uv sync --all-extras
```

This automatically creates a `.venv` directory. In your notebook, select the kernel named `teluxmmu-research`.