from setuptools import setup
from Cython.Build import cythonize
import os

# لیست تمام فایل‌هایی که باید کامپایل بشن (فقط فایل‌هایی که import می‌شن و اجرا دارن)
cython_modules = [
    "genetic_algoritm.pyx",
    "prepare_data_for_train.pyx",
    "threshold_finder.py",
    "model_pipeline.py",
    "drift_checker.py",
    "clear_data.py",
    "custom_indicators.py",
    "CustomKSTIndicator.py",
    "CustomVortexIndicator.py",
    "CustomIchimokuIndicator.py",
    "CustomWilliamsRIndicator.py",
    "CustomVolumeRateOfChangeIndicator.py",
    "CustomPivotPointIndicator.py",
    "CustomCandlestickPattern.py"
    # ❌ اینجا نباید numba_utils.py اضافه شده باشه
]


# اگر فایل pyx نیست، تبدیلش کن به .pyx (برای سازگاری)
for i, f in enumerate(cython_modules):
    if f.endswith(".py"):
        pyx = f.replace(".py", ".pyx")
        if os.path.exists(pyx):
            cython_modules[i] = pyx

setup(
    name="gold_project7_binary_build",
    ext_modules=cythonize(
        cython_modules,
        compiler_directives={'language_level': "3"}
    ),
)
