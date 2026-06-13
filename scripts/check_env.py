"""
환경 및 GPU 상태 확인 스크립트.

Usage:
    python scripts/check_env.py
"""

import sys


def section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def check_python():
    section("Python")
    print(f"  버전     : {sys.version.split()[0]}")
    print(f"  경로     : {sys.executable}")


def check_torch():
    section("PyTorch & CUDA")
    try:
        import torch
        print(f"  PyTorch  : {torch.__version__}")
        print(f"  CUDA 지원: {torch.cuda.is_available()}")

        if not torch.cuda.is_available():
            print("\n  ⚠ CUDA를 사용할 수 없습니다.")
            print("    → PyTorch CUDA 버전이 올바르게 설치됐는지 확인하세요.")
            print("    → pip install torch --index-url https://download.pytorch.org/whl/cu124")
            return

        print(f"  CUDA 버전: {torch.version.cuda}")
        print(f"  cuDNN    : {torch.backends.cudnn.version()}")
        print(f"  GPU 수   : {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / 1024 ** 3
            print(f"\n  [GPU {i}] {props.name}")
            print(f"    VRAM 전체    : {total_gb:.1f} GB")
            print(f"    Compute Cap  : {props.major}.{props.minor}")
            print(f"    SM 수        : {props.multi_processor_count}")

    except ImportError:
        print("  ✗ PyTorch 미설치")


def check_vram():
    section("VRAM 현재 사용량")
    try:
        import torch
        if not torch.cuda.is_available():
            print("  CUDA 불가 — 스킵")
            return

        for i in range(torch.cuda.device_count()):
            props  = torch.cuda.get_device_properties(i)
            total  = props.total_memory / 1024 ** 3
            reserv = torch.cuda.memory_reserved(i)  / 1024 ** 3
            alloc  = torch.cuda.memory_allocated(i) / 1024 ** 3
            free   = total - reserv
            print(f"  GPU {i}: {props.name}")
            print(f"    전체  : {total:.1f} GB")
            print(f"    사용중: {reserv:.1f} GB  (allocated {alloc:.1f} GB)")
            print(f"    여유  : {free:.1f} GB")
    except Exception as e:
        print(f"  오류: {e}")


def check_packages():
    section("주요 패키지")
    packages = [
        ("torch",           "PyTorch"),
        ("transformers",    "Transformers"),
        ("peft",            "PEFT"),
        ("datasets",        "Datasets"),
        ("accelerate",      "Accelerate"),
        ("tokenizers",      "Tokenizers"),
        ("huggingface_hub", "HuggingFace Hub"),
        ("numpy",           "NumPy"),
        ("rouge_score",     "rouge-score"),
        ("tqdm",            "tqdm"),
        ("yaml",            "PyYAML"),
    ]
    for mod_name, display in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  ✓  {display:<20}: {ver}")
        except ImportError:
            print(f"  ✗  {display:<20}: 미설치")
        except Exception as e:
            print(f"  !  {display:<20}: 임포트 오류 → {e}")


def check_hf_token():
    section("HuggingFace 토큰")
    try:
        from huggingface_hub import get_token
        token = get_token()
        if token:
            masked = token[:8] + "..." + token[-4:]
            print(f"  상태  : 로그인됨")
            print(f"  토큰  : {masked}")
        else:
            print("  상태  : 토큰 미설정")
            print("  설정법: huggingface-cli login")
    except Exception as e:
        print(f"  오류  : {e}")


def check_bf16():
    section("bfloat16 지원 여부 (RTX 4070 Ti 필수)")
    try:
        import torch
        if torch.cuda.is_available():
            supported = torch.cuda.is_bf16_supported()
            mark = "✓" if supported else "✗"
            print(f"  {mark} bfloat16: {'지원됨' if supported else '미지원 — fp16으로 대체 필요'}")
    except Exception as e:
        print(f"  오류: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("  noisy-llm-finetuning 환경 점검")
    print("=" * 50)

    check_python()
    check_torch()
    check_vram()
    check_packages()
    check_hf_token()
    check_bf16()

    print(f"\n{'='*50}\n")
