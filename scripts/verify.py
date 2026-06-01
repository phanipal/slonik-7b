from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False

ROOT = Path(__file__).resolve().parent.parent
WIN_PROJECT_HINT = "D:\\AI\\Projects\\slonik-7b"
WSL_RUNTIME_HINT = "~/slonik-runtime"


def _load_env_file_into_environ() -> None:
    env_path = ROOT / ".env"
    try:
        if not env_path.exists():
            return
    except OSError:
        return
    try:
        text = env_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_env_file_into_environ()

_uname = platform.uname()
IS_WINDOWS = platform.system() == "Windows"
IS_WSL = (
    platform.system() == "Linux"
    and ("microsoft" in _uname.release.lower() or "wsl" in _uname.release.lower())
)
IS_LINUX = platform.system() == "Linux" and not IS_WSL

PLATFORM = "Windows" if IS_WINDOWS else "WSL2" if IS_WSL else "Linux" if IS_LINUX else "macOS/other"

results: list[tuple[str, str, str, str]] = []


def safe_exists(path: Path) -> bool | None:
    try:
        return path.exists()
    except OSError:
        return None


def safe_is_dir(path: Path) -> bool | None:
    try:
        return path.is_dir()
    except OSError:
        return None


def safe_is_symlink(path: Path) -> bool | None:
    try:
        return path.is_symlink()
    except OSError:
        return None


def safe_readlink(path: Path) -> str | None:
    try:
        return os.readlink(path)
    except OSError:
        return None


def add(category: str, name: str, status: str, detail: str = "") -> None:
    results.append((category, name, status, detail))


def ok(category: str, name: str, detail: str = "") -> None:
    add(category, name, "PASS", detail)


def warn(category: str, name: str, detail: str = "") -> None:
    add(category, name, "WARN", detail)


def fail(category: str, name: str, detail: str = "") -> None:
    add(category, name, "FAIL", detail)


def check_file(category: str, label: str, path: Path, required: bool = True) -> None:
    existence = safe_exists(path)
    if existence is None:
        warn(category, label, f"unreadable from this OS (likely WSL symlink): {path}")
    elif existence:
        ok(category, label, str(path))
    elif required:
        fail(category, label, f"missing: {path}")
    else:
        warn(category, label, f"missing: {path}")


def check_dir(category: str, label: str, path: Path, required: bool = True) -> None:
    is_dir = safe_is_dir(path)
    if is_dir is None:
        warn(category, label, f"unreadable from this OS (likely WSL symlink): {path}")
    elif is_dir:
        ok(category, label, str(path))
    elif required:
        fail(category, label, f"missing dir: {path}")
    else:
        warn(category, label, f"missing dir: {path}")


def check_symlink(label: str, path: Path) -> None:
    is_link = safe_is_symlink(path)
    if is_link is None:
        warn("symlinks", label, f"unreadable from this OS: {path}")
        return
    if not is_link:
        existence = safe_exists(path)
        if existence is False:
            warn("symlinks", label, f"missing: {path}")
        else:
            warn("symlinks", label, f"not a symlink (plain dir): {path}")
        return
    target = safe_readlink(path)
    try:
        resolved = path.resolve()
        resolved_exists = resolved.exists()
    except OSError:
        resolved_exists = False
        resolved = None
    if resolved_exists:
        ok("symlinks", label, f"{path} -> {target}")
    else:
        fail("symlinks", label, f"dangling: {path} -> {target}")


def check_writable(category: str, label: str, path: Path) -> None:
    existence = safe_exists(path)
    if existence is None:
        warn(category, label, f"unreadable from this OS: {path}")
        return
    if not existence:
        fail(category, label, f"missing: {path}")
        return
    probe = path / ".write_probe"
    try:
        probe.write_text("ok")
        probe.unlink()
        ok(category, label, f"writable: {path}")
    except OSError as e:
        fail(category, label, f"not writable: {path} ({e})")


def env_var(category: str, name: str, required: bool = False) -> None:
    val = os.environ.get(name, "")
    if val:
        ok(category, name, f"len={len(val)}")
    elif required:
        fail(category, name, "not set")
    else:
        warn(category, name, "not set (optional)")


def parse_env_file(env_path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not safe_exists(env_path):
        return parsed
    try:
        text = env_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return parsed
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        parsed[k.strip()] = v.strip().strip('"').strip("'")
    return parsed


def check_env_file() -> None:
    env_path = ROOT / ".env"
    existence = safe_exists(env_path)
    if existence is None:
        warn(".env", ".env file", "unreadable from this OS")
        return
    if not existence:
        fail(".env", ".env file", f"missing: {env_path}")
        return
    parsed = parse_env_file(env_path)
    required = ["HF_TOKEN", "HF_USERNAME"]
    optional = ["WANDB_API_KEY", "WANDB_PROJECT", "ANTHROPIC_API_KEY",
                "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
    for k in required:
        if parsed.get(k):
            ok(".env", k, f"set (len={len(parsed[k])})")
        else:
            fail(".env", k, "missing or empty in .env")
    for k in optional:
        if parsed.get(k):
            ok(".env", k, f"set (len={len(parsed[k])})")
        else:
            warn(".env", k, "not set (optional)")


def check_python_version() -> None:
    v = sys.version_info
    label = f"Python {v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and v.minor == 11:
        ok("python", "version", label)
    elif v.major == 3 and v.minor in (10, 12):
        warn("python", "version", f"{label} (project tested on 3.11)")
    else:
        fail("python", "version", f"{label} — need 3.11.x")


def check_venv() -> None:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        ok("python", "venv", sys.prefix)
    else:
        fail("python", "venv", "not running inside a venv — activate first")


def import_check(name: str, alias: str | None = None) -> tuple[bool, str]:
    target = alias or name
    try:
        mod = importlib.import_module(target)
        version = getattr(mod, "__version__", "unknown")
        return True, version
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_packages() -> None:
    required = [
        ("torch", None),
        ("torchvision", None),
        ("transformers", None),
        ("trl", None),
        ("peft", None),
        ("accelerate", None),
        ("datasets", None),
        ("bitsandbytes", None),
        ("safetensors", None),
        ("huggingface_hub", None),
        ("fastapi", None),
        ("pydantic", None),
        ("sqlglot", None),
        ("pglast", None),
        ("click", None),
        ("rich", None),
        ("loguru", None),
        ("yaml", "yaml"),
        ("dotenv", "dotenv"),
        ("httpx", None),
        ("tqdm", None),
    ]
    linux_only = [
        ("triton", None),
        ("unsloth", None),
        ("unsloth_zoo", None),
        ("vllm", None),
        ("xformers", None),
    ]
    optional = [
        ("flash_attn", None),
        ("langfuse", None),
        ("wandb", None),
        ("streamlit", None),
        ("gradio", None),
    ]
    for name, alias in required:
        ok_, info = import_check(name, alias)
        (ok if ok_ else fail)("packages", name, info)
    if not IS_WINDOWS:
        for name, alias in linux_only:
            ok_, info = import_check(name, alias)
            (ok if ok_ else fail)("packages", name, info)
    for name, alias in optional:
        ok_, info = import_check(name, alias)
        (ok if ok_ else warn)("packages-optional", name, info)


def check_cuda() -> None:
    if IS_WINDOWS:
        warn("cuda", "skipped on Windows", "training runs in WSL")
        return
    try:
        import torch
    except ImportError:
        fail("cuda", "torch import", "torch not installed")
        return

    ok("cuda", "torch build", torch.__version__)
    if not torch.cuda.is_available():
        fail("cuda", "cuda.is_available()", "CUDA not available — check driver + cu128 wheel")
        return
    ok("cuda", "cuda.is_available()", "True")

    cuda_build = torch.version.cuda
    ok("cuda", "torch.version.cuda", cuda_build or "unknown")

    if cuda_build and not cuda_build.startswith("12.8"):
        warn("cuda", "cuda version", f"torch built for {cuda_build}; expected 12.8 for Blackwell")

    n = torch.cuda.device_count()
    ok("cuda", "device_count", str(n))

    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cc = torch.cuda.get_device_capability(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
        detail = f"{name} | cc={cc} | {total_mem:.1f} GB"
        if "5080" in name and cc == (12, 0):
            ok("cuda", f"gpu[{i}]", detail)
        elif cc == (12, 0):
            ok("cuda", f"gpu[{i}]", detail + " (Blackwell)")
        else:
            warn("cuda", f"gpu[{i}]", detail)


def check_disk_space() -> None:
    candidates = [ROOT]
    if not IS_WINDOWS:
        runtime = Path.home() / "slonik-runtime"
        if safe_exists(runtime):
            candidates.append(runtime)
        hf_cache = Path.home() / ".cache" / "huggingface"
        if safe_exists(hf_cache):
            candidates.append(hf_cache)
    for p in candidates:
        try:
            usage = shutil.disk_usage(p)
            free_gb = usage.free / 1024 ** 3
            total_gb = usage.total / 1024 ** 3
            detail = f"{free_gb:.1f} GB free / {total_gb:.1f} GB total at {p}"
            if free_gb < 30:
                fail("disk", str(p), detail + " (need ~80 GB for full pipeline)")
            elif free_gb < 80:
                warn("disk", str(p), detail)
            else:
                ok("disk", str(p), detail)
        except OSError as e:
            warn("disk", str(p), str(e))


def check_huggingface() -> None:
    if IS_WINDOWS:
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        warn("network", "huggingface auth", "HF_TOKEN not set")
        return
    try:
        import httpx
        r = httpx.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            ok("network", "huggingface auth", f"user={data.get('name')}")
        else:
            fail("network", "huggingface auth", f"HTTP {r.status_code}: {r.text[:100]}")
    except Exception as e:
        fail("network", "huggingface auth", str(e)[:120])


def check_dns(host: str = "huggingface.co") -> None:
    try:
        socket.gethostbyname(host)
        ok("network", f"DNS {host}", "resolved")
    except OSError as e:
        fail("network", f"DNS {host}", str(e))


def check_project_layout() -> None:
    expected_files = [
        "README.md", "requirements.txt", "pyproject.toml", ".env",
        "configs/sft_qlora.yaml", "configs/grpo.yaml",
        "configs/vllm_serve.yaml", "configs/datasets.yaml",
        "src/slonik/__init__.py",
        "src/slonik/training/sft.py", "src/slonik/training/grpo.py",
        "src/slonik/training/rewards.py", "src/slonik/training/exec_sandbox.py",
        "src/slonik/data/chatml.py", "src/slonik/data/prepare_bird.py",
        "src/slonik/eval/bird_runner.py",
        "src/slonik/serve/api.py",
        "scripts/download_data.py", "scripts/train_sft.py",
        "scripts/train_grpo.py", "scripts/eval_full.py",
    ]
    for rel in expected_files:
        check_file("layout", rel, ROOT / rel)

    for rel in ["data", "checkpoints", "outputs"]:
        path = ROOT / rel
        if IS_WSL or IS_LINUX:
            check_symlink(rel, path)
            if safe_exists(path):
                check_writable("io", rel, path)
        else:
            existence = safe_exists(path)
            if existence is None:
                warn("symlinks", rel, f"Linux symlink — invisible from Windows. Verify from WSL.")
            elif existence:
                if safe_is_symlink(path):
                    warn("symlinks", rel, "looks like a symlink — verify from WSL")
                else:
                    ok("layout", rel, str(path))
            else:
                warn("layout", rel, f"missing: {path}")


def check_wsl_runtime() -> None:
    if IS_WINDOWS:
        return
    runtime = Path.home() / "slonik-runtime"
    check_dir("wsl runtime", "~/slonik-runtime", runtime)
    for sub in ("data", "checkpoints", "outputs", ".venv-linux"):
        check_dir("wsl runtime", f"~/slonik-runtime/{sub}", runtime / sub, required=False)


def check_nvidia_smi() -> None:
    if IS_WINDOWS:
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            fail("nvidia", "nvidia-smi", out.stderr.strip()[:120])
            return
        for line in out.stdout.strip().splitlines():
            ok("nvidia", "gpu info", line)
    except FileNotFoundError:
        fail("nvidia", "nvidia-smi", "not found (driver / WSL passthrough missing)")
    except subprocess.TimeoutExpired:
        warn("nvidia", "nvidia-smi", "timeout")


def check_yaml_configs() -> None:
    configs = [
        "configs/sft_qlora.yaml",
        "configs/grpo.yaml",
        "configs/vllm_serve.yaml",
        "configs/datasets.yaml",
    ]
    try:
        import yaml
    except ImportError:
        warn("configs", "yaml parse", "pyyaml not installed")
        return
    for rel in configs:
        path = ROOT / rel
        if not safe_exists(path):
            warn("configs", rel, "missing (already reported in layout)")
            continue
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
            ok("configs", rel, "parses cleanly")
        except yaml.YAMLError as e:
            fail("configs", rel, f"YAML parse error: {str(e)[:120]}")
        except OSError as e:
            fail("configs", rel, str(e))


def check_project_imports() -> None:
    if IS_WINDOWS:
        return
    src_path = ROOT / "src"
    if not safe_exists(src_path):
        return
    sys.path.insert(0, str(src_path))
    modules = [
        "slonik.data.chatml",
        "slonik.data.schema",
        "slonik.training.rewards",
        "slonik.training.exec_sandbox",
    ]
    for m in modules:
        try:
            importlib.import_module(m)
            ok("project imports", m, "OK")
        except Exception as e:
            fail("project imports", m, f"{type(e).__name__}: {str(e)[:120]}")


def check_cuda_compute() -> None:
    if IS_WINDOWS:
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    try:
        a = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        c = a @ b
        torch.cuda.synchronize()
        ok("cuda compute", "bf16 matmul", f"shape={tuple(c.shape)} dtype={c.dtype}")
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        fail("cuda compute", "bf16 matmul", f"{type(e).__name__}: {str(e)[:160]}")


def check_bitsandbytes_runtime() -> None:
    if IS_WINDOWS:
        return
    try:
        out = subprocess.run(
            [sys.executable, "-m", "bitsandbytes"],
            capture_output=True, text=True, timeout=30,
        )
        text = (out.stdout + out.stderr).lower()
        if out.returncode == 0 and ("library" in text or "cuda" in text) and "error" not in text:
            ok("bitsandbytes", "self-check", "CUDA libs found")
        elif "library" in text and "could not be loaded" in text:
            fail("bitsandbytes", "self-check", "library not loadable — check CUDA install")
        else:
            warn("bitsandbytes", "self-check", out.stdout.strip()[-200:] or out.stderr.strip()[-200:])
    except FileNotFoundError:
        fail("bitsandbytes", "self-check", "module not installed")
    except subprocess.TimeoutExpired:
        warn("bitsandbytes", "self-check", "timeout (>30s)")
    except Exception as e:
        warn("bitsandbytes", "self-check", str(e)[:160])


def check_triton_kernel() -> None:
    if IS_WINDOWS:
        return
    try:
        import triton
    except ImportError:
        return
    try:
        import torch
        if not torch.cuda.is_available():
            return
        ok("triton", "import + cuda visible", f"version={triton.__version__}")
    except Exception as e:
        warn("triton", "probe", str(e)[:160])


def check_wsl_version() -> None:
    if not IS_WSL:
        return
    try:
        proc_version = Path("/proc/version").read_text()
        if "WSL2" in proc_version or "microsoft-standard" in proc_version.lower():
            ok("wsl", "WSL version", "WSL2")
        elif "Microsoft" in proc_version and "WSL2" not in proc_version:
            fail("wsl", "WSL version", "WSL1 detected — need WSL2 for CUDA passthrough. Run 'wsl --set-version <distro> 2'")
        else:
            warn("wsl", "WSL version", "indeterminate")
    except OSError as e:
        warn("wsl", "WSL version", str(e))


def check_nvidia_driver_version() -> None:
    if IS_WINDOWS:
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return
        version = out.stdout.strip().splitlines()[0]
        major = int(version.split(".")[0])
        if major >= 555:
            ok("nvidia", "driver version", f"{version} (Blackwell-capable)")
        elif major >= 535:
            warn("nvidia", "driver version", f"{version} (works for Ampere/Ada; update to 555+ for full Blackwell)")
        else:
            fail("nvidia", "driver version", f"{version} too old — RTX 5080 needs 555+")
    except (FileNotFoundError, ValueError, IndexError, subprocess.TimeoutExpired) as e:
        warn("nvidia", "driver version", str(e)[:100])


def check_hf_cache() -> None:
    if IS_WINDOWS:
        return
    cache = Path(os.environ.get("HF_HOME", "")) if os.environ.get("HF_HOME") else (Path.home() / ".cache" / "huggingface")
    try:
        cache.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(cache)
        free_gb = usage.free / 1024 ** 3
        detail = f"{cache} | {free_gb:.1f} GB free"
        if free_gb < 20:
            fail("hf cache", "free space", detail + " — need ~14 GB for base model + ~14 GB for merged")
        elif free_gb < 40:
            warn("hf cache", "free space", detail)
        else:
            ok("hf cache", "free space", detail)
    except OSError as e:
        warn("hf cache", "writable", str(e))


def check_system_time() -> None:
    try:
        import httpx
        r = httpx.head("https://huggingface.co", timeout=5)
        srv_date = r.headers.get("date")
        if not srv_date:
            warn("system", "clock vs network", "no Date header from huggingface.co")
            return
        from email.utils import parsedate_to_datetime
        from datetime import datetime, timezone
        srv_dt = parsedate_to_datetime(srv_date)
        local_dt = datetime.now(timezone.utc)
        skew = abs((local_dt - srv_dt).total_seconds())
        if skew > 300:
            fail("system", "clock skew vs huggingface.co", f"{skew:.0f}s — TLS will fail. Sync your system clock.")
        elif skew > 60:
            warn("system", "clock skew vs huggingface.co", f"{skew:.0f}s")
        else:
            ok("system", "clock skew vs huggingface.co", f"{skew:.0f}s")
    except Exception as e:
        warn("system", "clock vs network", str(e)[:120])


def check_git_installed() -> None:
    try:
        out = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            ok("tools", "git", out.stdout.strip())
        else:
            warn("tools", "git", "git found but errored")
    except FileNotFoundError:
        fail("tools", "git", "not installed (needed for GGUF/llama.cpp conversion step)")
    except subprocess.TimeoutExpired:
        warn("tools", "git", "timeout")


def check_api_wandb() -> None:
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        return
    try:
        import base64
        import httpx
        auth = base64.b64encode(f"api:{key}".encode()).decode()
        r = httpx.post(
            "https://api.wandb.ai/graphql",
            headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
            json={"query": "{ viewer { username } }"},
            timeout=10,
        )
        if r.status_code == 200 and "viewer" in r.text:
            data = r.json().get("data", {}).get("viewer", {})
            ok("apis", "W&B auth", f"user={data.get('username', 'unknown')}")
        else:
            fail("apis", "W&B auth", f"HTTP {r.status_code}: {r.text[:120]}")
    except Exception as e:
        fail("apis", "W&B auth", str(e)[:160])


def check_api_anthropic() -> None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    base = os.environ.get("ANTHROPIC_BASE_URL", "")
    if not key:
        return
    target = "DeepSeek" if "deepseek" in base.lower() else "Anthropic"
    model = os.environ.get("SYNTH_MODEL") or ("deepseek-v4-pro" if "deepseek" in base.lower() else "claude-haiku-4-5-20251001")
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=key, base_url=base or None, timeout=10.0)
        msg = client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        )
        text_parts = []
        for block in msg.content:
            t = getattr(block, "text", None) or getattr(block, "thinking", None)
            if t:
                text_parts.append(t)
        text = " ".join(text_parts).strip()
        usage = getattr(msg, "usage", None)
        usage_str = f" tokens(in={getattr(usage, 'input_tokens', '?')}, out={getattr(usage, 'output_tokens', '?')})" if usage else ""
        if text:
            ok("apis", f"{target} auth", f"model={model} reply='{text[:40]}'{usage_str}")
        else:
            ok("apis", f"{target} auth", f"model={model} responded (no text block){usage_str}")
    except ImportError:
        warn("apis", f"{target} auth", "anthropic SDK not installed")
    except Exception as e:
        fail("apis", f"{target} auth", str(e)[:200])


def check_api_langfuse() -> None:
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if not pk or not sk:
        return
    try:
        import httpx
        r = httpx.get(f"{host}/api/public/health", timeout=5)
        if r.status_code == 200:
            ok("apis", "Langfuse host", host)
        else:
            warn("apis", "Langfuse host", f"HTTP {r.status_code}")
    except Exception as e:
        warn("apis", "Langfuse host", str(e)[:120])


def check_api_hf_dataset_reachable() -> None:
    try:
        import httpx
        r = httpx.head(
            "https://huggingface.co/datasets/xlangai/spider",
            timeout=10, follow_redirects=True,
        )
        if r.status_code < 400:
            ok("apis", "HF dataset reachable", f"HTTP {r.status_code}")
        else:
            warn("apis", "HF dataset reachable", f"HTTP {r.status_code}")
    except Exception as e:
        warn("apis", "HF dataset reachable", str(e)[:120])


def print_remediation() -> None:
    fails = [(c, n, d) for c, n, s, d in results if s == "FAIL"]
    if not fails:
        return
    header = "REMEDIATION HINTS"
    if HAS_RICH:
        console.print(f"\n[bold red]{header}[/]")
    else:
        print(f"\n{header}\n" + "-" * len(header))
    for c, n, d in fails:
        line = f"  [{c}] {n}: {d}"
        if HAS_RICH:
            console.print(line)
        else:
            print(line)



def render() -> int:
    summary = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for *_, status, _ in [(c, n, s, d) for c, n, s, d in results]:
        summary[status] = summary.get(status, 0) + 1

    if HAS_RICH:
        table = Table(title=f"Slonik environment check — {PLATFORM}")
        table.add_column("Category")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Detail", overflow="fold")
        for cat, name, status, detail in results:
            colour = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[status]
            table.add_row(cat, name, f"[{colour}]{status}[/]", detail)
        console.print(table)
        console.print(
            f"\n[green]PASS:[/] {summary['PASS']}  "
            f"[yellow]WARN:[/] {summary['WARN']}  "
            f"[red]FAIL:[/] {summary['FAIL']}"
        )
    else:
        print(f"\nSlonik environment check — {PLATFORM}\n" + "=" * 60)
        for cat, name, status, detail in results:
            print(f"[{status:4}] {cat:>16}  {name:<32}  {detail}")
        print("=" * 60)
        print(f"PASS: {summary['PASS']}   WARN: {summary['WARN']}   FAIL: {summary['FAIL']}")

    report_path = ROOT / "outputs" / "verify_report.json"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({
            "platform": PLATFORM,
            "python": sys.version,
            "summary": summary,
            "results": [
                {"category": c, "check": n, "status": s, "detail": d}
                for c, n, s, d in results
            ],
        }, indent=2))
    except OSError:
        pass

    return 0 if summary["FAIL"] == 0 else 1


def _run(label: str, func) -> None:
    print(f"  • {label} ...", end="", flush=True)
    try:
        func()
        print(" done", flush=True)
    except Exception as e:
        print(f" ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)


def main() -> int:
    print(f"Platform detected: {PLATFORM}")
    print(f"Project root: {ROOT}\n")
    print("Running checks (this can take 30-60s the first time):\n")

    _run("project layout", check_project_layout)
    _run(".env file", check_env_file)
    _run("YAML configs", check_yaml_configs)
    _run("python version", check_python_version)
    _run("venv", check_venv)
    _run("packages (this can take 20s as torch/transformers load)", check_packages)
    _run("project imports", check_project_imports)
    _run("CUDA visibility", check_cuda)
    _run("CUDA bf16 matmul", check_cuda_compute)
    _run("nvidia-smi", check_nvidia_smi)
    _run("nvidia driver version", check_nvidia_driver_version)
    _run("bitsandbytes self-check", check_bitsandbytes_runtime)
    _run("triton probe", check_triton_kernel)
    _run("WSL runtime dirs", check_wsl_runtime)
    _run("WSL version", check_wsl_version)
    _run("HF cache space", check_hf_cache)
    _run("disk space", check_disk_space)
    _run("DNS", check_dns)
    _run("system clock", check_system_time)
    _run("git", check_git_installed)
    _run("HF auth (API call)", check_huggingface)
    _run("HF dataset reachable", check_api_hf_dataset_reachable)
    _run("W&B auth", check_api_wandb)
    _run("Anthropic/DeepSeek auth (sends 1 short msg)", check_api_anthropic)
    _run("Langfuse host", check_api_langfuse)

    print()
    exit_code = render()
    print_remediation()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
