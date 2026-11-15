import sqlite3

# 1. יצירה או חיבור למסד נתונים מקומי בשם substitutes.db
conn = sqlite3.connect("substitutes.db")
cursor = conn.cursor()

# 2. יצירת טבלת סייעות
cursor.execute("""
CREATE TABLE IF NOT EXISTS substitutes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    distance_km REAL,
    experience_years INTEGER
)
""")

# 3. הוספת נתוני דוגמה של סייעות (לפי תרחיש 1 שלנו)
sample_substitutes = [
    ("שרה כהן", "ירושלים", 1.0, 3),
    ("לאה גולד", "ירושלים", 5.0, 1),
    ("רחל אברהם", "ירושלים", 2.0, 5),
    ("רבקה שוורץ", "ירושלים", 8.0, 2)
]

# מכניסים את הנתונים רק אם הטבלה ריקה כדי לא ליצור כפילויות
cursor.execute("SELECT COUNT(*) FROM substitutes")
if cursor.fetchone()[0] == 0:
    cursor.executemany("""
    INSERT INTO substitutes (name, city, distance_km, experience_years)
    VALUES (?, ?, ?, ?)
    """, sample_substitutes)
    conn.commit()

print("מסד הנתונים הוקם ונתוני הדוגמה הוזנו בהצלחה!\n")

# 4. הדמיה של שליפה וחישוב יוריסטיקה (דירוג משוקלל)
# נניח שחיפשנו סייעות לגן מסוים, ונרצה למיין לפי נוסחה: ככל שהמרחק קטן יותר והניסיון רב יותר - הציון גבוה יותר.
print("--- תוצאות דירוג הסייעות עבור הקריאה ---")
cursor.execute("SELECT name, distance_km, experience_years FROM substitutes")
substitutes = cursor.fetchall()

scored_substitutes = []
for sub in substitutes:
    name, distance, experience = sub
    # נוסחה פשוטה ליוריסטיקה: ניסיון כפול 10 פחות מרחק בקילומטרים
    score = (experience * 10) - (distance * 2)
    scored_substitutes.append((name, score, distance, experience))

# מיון מהציון הגבוה לנמוך
scored_substitutes.sort(key=lambda x: x[1], reverse=True)

for i, sub in enumerate(scored_substitutes, 1):
    print(f"{i}. {sub[0]} | ציון התאמה: {sub[1]} (מרחק: {sub[2]}ק\"מ, ניסיון: {sub[3]} שנים)")

conn.close()