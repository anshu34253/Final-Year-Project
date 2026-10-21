import os
import sqlite3

user_profile = os.environ.get('USERPROFILE', '')
chrome_cookies = os.path.join(user_profile, 'AppData', 'Local', 'Google', 'Chrome', 'User Data', 'Default', 'Network', 'Cookies')
edge_cookies = os.path.join(user_profile, 'AppData', 'Local', 'Microsoft', 'Edge', 'User Data', 'Default', 'Network', 'Cookies')

def test_read_sqlite(db_path, name):
    if not os.path.exists(db_path):
        print(name, 'path does not exist')
        return
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro&immutable=1', uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT host_key, name FROM cookies WHERE host_key LIKE '%youtube%' LIMIT 10")
        rows = cursor.fetchall()
        print(f"SUCCESS: {name} YouTube cookies count = {len(rows)}")
        for r in rows[:3]:
            print("  Cookie:", r)
        conn.close()
    except Exception as e:
        print(f"FAILED: {name} read error = {e}")

test_read_sqlite(chrome_cookies, 'Chrome')
test_read_sqlite(edge_cookies, 'Edge')
