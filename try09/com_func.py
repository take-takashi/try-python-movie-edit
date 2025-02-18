import sys
import os

def resource_path(relative_path: str) -> str:
    """
    指定されたリソースファイルやディレクトリの絶対パスを返します。
    開発環境では現在の作業ディレクトリを基準に、PyInstallerでビルドした場合は
    sys._MEIPASSで指定された一時ディレクトリを基準にパスを組み立てます。

    Parameters:
        relative_path (str): リソースの相対パス

    Returns:
        str: リソースの絶対パス
    """
    base_path = getattr(sys, '_MEIPASS', os.path.abspath('.'))
    return os.path.join(base_path, relative_path)