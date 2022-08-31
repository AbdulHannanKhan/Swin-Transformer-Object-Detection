def window_partition(x, window_size, channel_last=True):
    """
    Args:
        x: (B, W, H, C)
        window_size (int): window size
    Returns:
        windows: (B, num_windows, window_size * window_size, C)
        :param channel_last: if channel is last dim
    """
    if not channel_last:
        x = x.permute(0, 2, 3, 1)
    B, W, H, C = x.shape
    x = x.view(B, W // window_size, window_size, H // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, W, H):
    """
    Args:
        windows: (B, num_windows, window_size*window_size, C)
        window_size (int): Window size
        W (int): Width of image
        H (int): Height of image
    Returns:
        x: (B, C, W, H)
    """
    B = windows.shape[0]
    x = windows.reshape(B, W // window_size, H // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, W, H)
    return x