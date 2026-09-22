"""Loader rozszerzenia CUDA z kernelem zapisu nvfp4_ds_mla.

Instalowany do dist-packages; .so budowane w Dockerfile do /opt/nvfp4ext.
"""
import glob
import importlib.util

import torch  # noqa: F401  (laduje libc10/libtorch przed .so)

_so = glob.glob("/opt/nvfp4ext/nvfp4_cache_ext/*.so")
if not _so:
    raise ImportError("brak zbudowanego nvfp4_cache_ext w /opt/nvfp4ext")
_spec = importlib.util.spec_from_file_location("nvfp4_cache_ext", _so[0])
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

concat_and_cache_nvfp4_ds_mla = _mod.concat_and_cache_nvfp4_ds_mla
