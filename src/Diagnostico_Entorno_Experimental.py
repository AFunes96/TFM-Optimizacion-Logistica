import importlib.metadata
import platform
import subprocess
import sys

def ejecutar_powershell(comando):
    try:
        return subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", comando],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""

cpu = ejecutar_powershell("(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)")
ram_bytes = ejecutar_powershell("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")

if not cpu:
    cpu = platform.processor()

try:
    ram_gb = float(ram_bytes) / (1024 ** 3)
    ram = f"{ram_gb:.1f} GB"
except Exception:
    ram = "No detectada"

try:
    pulp_version = importlib.metadata.version("pulp")
except Exception:
    pulp_version = "No detectada"

try:
    highspy_version = importlib.metadata.version("highspy")
except Exception:
    highspy_version = "No detectada"

try:
    from highspy import Highs
    highs_version = Highs().version()
except Exception:
    highs_version = highspy_version

lineas = [
    f"Procesador: {cpu}",
    f"RAM: {ram}",
    f"Sistema operativo: {platform.platform()}",
    f"Python: {sys.version.split()[0]}",
    f"PuLP: {pulp_version}",
    f"highspy: {highspy_version}",
    f"HiGHS: {highs_version}"
]

resultado = "\n".join(lineas)
print(resultado)

Path = __import__("pathlib").Path
Path("entorno_experimental.txt").write_text(resultado, encoding="utf-8")
