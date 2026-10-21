import os
import sys
import ctypes

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
FILE_SHARE_DELETE = 0x00000004
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80

def copy_locked_file_win32(src_path, dst_path):
    kernel32 = ctypes.windll.kernel32
    
    handle = kernel32.CreateFileW(
        src_path,
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        None,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        None
    )
    
    INVALID_HANDLE_VALUE = -1
    if handle == INVALID_HANDLE_VALUE or handle == 0 or handle == 0xFFFFFFFF:
        err = kernel32.GetLastError()
        raise OSError(f"CreateFileW failed with error code {err}")
        
    try:
        buf = ctypes.create_string_buffer(64 * 1024)
        bytes_read = ctypes.c_ulong(0)
        
        with open(dst_path, "wb") as f_out:
            while True:
                success = kernel32.ReadFile(handle, buf, len(buf), ctypes.byref(bytes_read), None)
                if not success or bytes_read.value == 0:
                    break
                f_out.write(buf.raw[:bytes_read.value])
    finally:
        kernel32.CloseHandle(handle)
        
    return os.path.exists(dst_path) and os.path.getsize(dst_path) > 0

user_profile = os.environ.get('USERPROFILE', '')
chrome_cookies = os.path.join(user_profile, 'AppData', 'Local', 'Google', 'Chrome', 'User Data', 'Default', 'Network', 'Cookies')
edge_cookies = os.path.join(user_profile, 'AppData', 'Local', 'Microsoft', 'Edge', 'User Data', 'Default', 'Network', 'Cookies')

print("Testing win32 copy on Chrome cookies...")
try:
    res = copy_locked_file_win32(chrome_cookies, "data/chrome_cookies_temp.db")
    print("Chrome copy win32 result:", res, "Size:", os.path.getsize("data/chrome_cookies_temp.db"))
except Exception as e:
    print("Chrome win32 copy failed:", e)

print("\nTesting win32 copy on Edge cookies...")
try:
    res = copy_locked_file_win32(edge_cookies, "data/edge_cookies_temp.db")
    print("Edge copy win32 result:", res, "Size:", os.path.getsize("data/edge_cookies_temp.db"))
except Exception as e:
    print("Edge win32 copy failed:", e)
