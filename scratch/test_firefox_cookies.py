import os
import glob
import sqlite3

user_profile = os.environ.get('USERPROFILE', '')
firefox_dir = os.path.join(user_profile, 'AppData', 'Roaming', 'Mozilla', 'Firefox', 'Profiles')

profiles = glob.glob(os.path.join(firefox_dir, '*'))
print("Firefox profiles found:", profiles)

for p in profiles:
    cookie_db = os.path.join(p, 'cookies.sqlite')
    if os.path.exists(cookie_db):
        print("\nFound Firefox cookie db:", cookie_db)
        try:
            conn = sqlite3.connect(cookie_db)
            cursor = conn.cursor()
            cursor.execute("SELECT host, name, value FROM moz_cookies WHERE host LIKE '%youtube%' LIMIT 10")
            rows = cursor.fetchall()
            print("FIREFOX YOUTUBE COOKIES COUNT:", len(rows))
            for r in rows[:5]:
                print("  Cookie:", r[0], r[1])
            conn.close()
        except Exception as e:
            print("Firefox sqlite read failed:", e)
