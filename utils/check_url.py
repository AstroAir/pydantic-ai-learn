from urllib.parse import urlparse


def ensure_trailing_slash(url: str) -> str:
    """
    检查输入是否为有效的 URL，如果是且路径不以 '/' 结尾，则在末尾添加 '/'。
    否则返回原字符串（或抛出异常，取决于需求）。

    参数:
        url (str): 待处理的 URL 字符串

    返回:
        str: 处理后的 URL（确保以 '/' 结尾）

    异常:
        ValueError: 如果输入不是有效的 URL
    """
    if not isinstance(url, str):
        raise ValueError("Input must be a string.")

    # 使用 urlparse 解析 URL
    parsed = urlparse(url)

    # 判断是否为有效 URL：至少要有 scheme 和 netloc
    if not all([parsed.scheme, parsed.netloc]):
        raise ValueError(f"'{url}' is not a valid URL.")

    # 检查路径是否以 '/' 结尾
    if not parsed.path.endswith("/"):
        # 重建 URL，确保路径以 '/' 结尾
        new_path = parsed.path + "/"
        # 使用 _replace 创建新的 ParseResult 对象
        new_parsed = parsed._replace(path=new_path)
        return new_parsed.geturl()
    return url
