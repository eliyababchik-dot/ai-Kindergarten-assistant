import sqlite3

# 1. התחברות למסד נתונים ויצירת טבלאות
conn = sqlite3.connect("substitutes.db")
cursor = conn.cursor()

# טבלת סייעות
cursor.execute("""
CREATE TABLE IF NOT EXISTS substitutes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    experience_years INTEGER
)
""")

# טבלת מרחקים בין סייעות לגנים שונים (כל סייעת והמרחק שלה לכל גן)
cursor.execute("""
CREATE TABLE IF NOT EXISTS substitute_distances (
    substitute_id INTEGER,
    garden_name TEXT,
    distance_km REAL,
    FOREIGN KEY (substitute_id) REFERENCES substitutes (id)
)
""")

# הנתונים לדוגמה (אם הטבלה ריקה)
cursor.execute("SELECT COUNT(*) FROM substitutes")
if cursor.fetchone()[0] == 0:
    # הכנסת סייעות
    cursor.executemany("INSERT INTO substitutes (name, experience_years) VALUES (?, ?)", [
        ("שרה כהן", 3),
        ("לאה גולד", 1),
        ("רחל אברהם", 5)
    ])
    
    # הגדרת מרחקים שונים לכל סייעת לכל גן!
    # סייעת 1 (שרה) קרובה לגן שקד אבל רחוקה מגן רימון
    # סייעת 2 (לאה) בינונית בשניהם
    # סייעת 3 (רחל) רחוקה מגן שקד אבל קרובה מאוד לגן רימון
    distances_data = [
        (1, "גן שקד", 1.0), (1, "גן רימון", 8.0),
        (2, "גן שקד", 4.0), (2, "גן רימון", 4.5),
        (3, "גן שקד", 7.0), (3, "גן רימון", 1.5)
    ]
    cursor.executemany("INSERT INTO substitute_distances (substitute_id, garden_name, distance_km) VALUES (?, ?, ?)", distances_data)
    conn.commit()

# ==========================================
# 2. הרכזת מדווחת על היעדרות בגן ספציפי!
# ==========================================
target_garden = "גן רימון"  # נסי להחליף ל- "גן שקד" ותראי איך הדירוג משתנה!
print(label := f"--- תוצאות דירוג הסייעות עבור דיווח היעדרות ב-{target_garden} ---")

# 3. שליפת הסייעות והמרחק שלהן *אך ורק* ביחס לגן הנבחר
query = """
SELECT s.name, s.experience_years, d.distance_km 
FROM substitutes s
JOIN substitute_distances d ON s.id = d.substitute_id
WHERE d.garden_name = ?
"""

cursor.execute(query, (target_garden,))
substitutes_at_garden = cursor.fetchall()

# 4. חישוב יוריסטיקה דינמית (ניסיון כפול 10 פחות המרחק מהגן הספציפי הזה)
scored_substitutes = []
for sub in substitutes_at_garden:
    name, experience, distance = sub
    score = (experience * 10) - (distance * 2)
    scored_substitutes.append((name, score, distance, experience))

# מיון מהציון הגבוה לנמוך
scored_substitutes.sort(key=lambda x: x[1], reverse=True)

# הצגת התוצאות לרכזת
for i, sub in enumerate(scored_substitutes, 1):
    print(f"{i}. {sub[0]} | ציון התאמה: {sub[1]} (מרחק מ{target_garden}: {sub[2]}ק\"מ, ניסיון: {sub[3]} שנים)")

conn.close()