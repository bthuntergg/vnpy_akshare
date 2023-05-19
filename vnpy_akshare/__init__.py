import importlib_metadata

from .akshre_feed import AKShareDataFeed as Datafeed


try:
    __version__ = importlib_metadata.version("vnpy_akshare")
except importlib_metadata.PackageNotFoundError:
    __version__ = "dev"
