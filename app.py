from flask import Flask, request, render_template_string
from transformers import pipeline
from deep_translator import GoogleTranslator
import sqlite3
import spacy
import requests
from bs4 import BeautifulSoup

# Flask 앱 초기화
app = Flask(__name__)

# spaCy 모델 로드
nlp = spacy.load("en_core_web_sm")

# HuggingFace 감정 분석 모델 로드
emotion_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# 감정 라벨 매핑
emotion_labels = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE"
}

# SQLite 데이터베이스 연결 및 테이블 생성
conn = sqlite3.connect("dreams.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS dreams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dream_text TEXT,
    keywords TEXT,
    emotion TEXT,
    confidence REAL,
    search_results TEXT
)
""")
conn.commit()

### Bing 검색 코드 ###
def search_dream_interpretation_bing(dream_text):
    """Bing에서 꿈 해석 검색 및 결과 반환"""
    query = f"{dream_text} 꿈 해석"
    search_url = f"https://www.bing.com/search?q={query}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        for result in soup.find_all("li", class_="b_algo"):
            snippet = result.find("p")
            if snippet:
                results.append(snippet.get_text(strip=True))

        return "<br>".join(results[:3]) if results else "검색 결과를 찾을 수 없습니다."
    except requests.exceptions.RequestException as e:
        return f"검색 요청 중 오류가 발생했습니다: {e}"

### NLP 코드 ###
def translate_to_english(text):
    """한국어 텍스트를 영어로 번역"""
    translator = GoogleTranslator(source="ko", target="en")
    return translator.translate(text)

def extract_keywords(text):
    """텍스트에서 키워드 추출"""
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ", "VERB", "ADV"]]

def analyze_emotion(text):
    """영어 텍스트 감정 분석"""
    emotion_result = emotion_analyzer(text)
    raw_label = emotion_result[0]['label']
    emotion = emotion_labels[raw_label]
    confidence = emotion_result[0]['score']
    return emotion, confidence

### Flask 라우팅 ###
@app.route('/')
def home():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RE:DREAM</title>
            <style>
                body {
                    font-family: 'Arial', sans-serif;
                    background: url("{{ url_for('static', filename='images/dreams_background.jpg') }}") no-repeat center center fixed;
                    background-size: cover;
                    color: white;
                    margin: 0;
                    padding: 0;
                }
                header {
                    background-color: rgba(0, 51, 102, 0.8);
                    color: white;
                    padding: 1rem 0;
                    text-align: center;
                    font-size: 2rem;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
                }
                main {
                    max-width: 600px;
                    margin: 2rem auto;
                    padding: 2rem;
                    background: rgba(255, 255, 255, 0.9);
                    border-radius: 15px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                    text-align: center;
                    color: #333;
                }
                label {
                    font-weight: bold;
                    display: block;
                    margin-bottom: 0.5rem;
                }
                textarea {
                    width: 100%;
                    padding: 1rem;
                    border-radius: 10px;
                    border: 1px solid #87CEEB;
                    font-size: 1rem;
                    margin-bottom: 1rem;
                    resize: none;
                }
                button, input[type="submit"] {
                    background-color: #00509E;
                    color: white;
                    border: none;
                    padding: 0.7rem 1.5rem;
                    border-radius: 10px;
                    cursor: pointer;
                    font-size: 1rem;
                    margin: 0.5rem 0;
                    transition: all 0.3s ease;
                }
                button:hover, input[type="submit"]:hover {
                    background-color: #003f7f;
                }
                footer {
                    text-align: center;
                    margin-top: 2rem;
                    font-size: 0.9rem;
                    color: white;
                }
            </style>
        </head>
        <body>
            <header>
                🌙 RE:DREAM
            </header>
            <main>
                <form method="post" action="/analyze">
                    <label for="dream">꿈 내용을 입력하세요:</label>
                    <textarea id="dream" name="dream_text" rows="4" required></textarea>
                    <input type="submit" value="분석하기">
                </form>
                <form action="/records" method="get">
                    <button type="submit">기록 보기</button>
                </form>
            </main>
            <footer>
                &copy; 2024 꿈 해석 서비스 | 꿈에서 미래를 찾다
            </footer>
        </body>
        </html>
    ''')


@app.route('/analyze', methods=['POST'])
def analyze():
    dream_text = request.form['dream_text']

    if dream_text:
        search_results = search_dream_interpretation_bing(dream_text)
        translated_text = translate_to_english(dream_text)
        emotion, confidence = analyze_emotion(translated_text)
        keywords = extract_keywords(translated_text)

        dream_data = (
            dream_text,
            ", ".join(keywords),
            emotion,
            confidence,
            search_results
        )
        cursor.execute(
            "INSERT INTO dreams (dream_text, keywords, emotion, confidence, search_results) VALUES (?, ?, ?, ?, ?)",
            dream_data
        )
        conn.commit()

        image_path = ""
        if emotion == "positive":
            image_path = "/static/image/긍정.png"
        elif emotion == "negative":
            image_path = "/static/image/부정.png"
        elif emotion == "neutral":
            image_path = "./static/image/중립.png"

        return render_template_string(f'''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>분석 결과</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background: #f0f8ff;
                        color: #333;
                        margin: 0;
                        padding: 0;
                    }}
                    header {{
                        background-color: #00509E;
                        color: white;
                        padding: 1rem;
                        text-align: center;
                        font-size: 1.5rem;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
                    }}
                    main {{
                        max-width: 800px;
                        margin: 2rem auto;
                        padding: 2rem;
                        background: white;
                        border-radius: 15px;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                        font-size: 1rem;
                    }}
                    h1 {{
                        color: #00509E;
                        text-align: center;
                        margin-bottom: 1rem;
                    }}
                    img {{
                        display: block;
                        margin: 0 auto 1.5rem auto;
                        width: 100%;
                        max-width: 300px;
                        border-radius: 15px;
                    }}
                    p {{
                        margin-bottom: 1rem;
                        line-height: 1.6;
                    }}
                    strong {{
                        color: #00509E;
                    }}
                    a {{
                        display: inline-block;
                        margin-top: 1rem;
                        padding: 0.5rem 1rem;
                        background-color: #00509E;
                        color: white;
                        text-decoration: none;
                        border-radius: 5px;
                        transition: all 0.3s ease;
                    }}
                    a:hover {{
                        background-color: #003f7f;
                    }}
                </style>
            </head>
            <body>
                <header>
                    분석 결과
                </header>
                <main>
                    <h1>🌟 꿈 분석 결과</h1>
                    <img src="static/images/중립.png">
                    <p><strong>꿈 내용:</strong> {dream_text}</p>
                    <p><strong>번역된 내용:</strong> {translated_text}</p>
                    <p><strong>감정:</strong> {emotion}</p>
                    <p><strong>확률:</strong> {confidence:.2f}</p>
                    <p><strong>추출된 키워드:</strong> {', '.join(keywords)}</p>
                    <p><strong>검색 결과:</strong><br>{search_results}</p>
                    <a href="/">다른 꿈 분석하기</a>
                    <a href="/records">기록 보기</a>
                </main>
            </body>
            </html>
        ''')

    else:
        return "꿈 내용을 입력하지 않았습니다."

@app.route('/records', methods=['GET'])
def records():
    cursor.execute("SELECT id, dream_text, emotion, search_results FROM dreams")
    records = cursor.fetchall()

    if records:
        records_html = "".join([
            f"<p><strong>ID:</strong> {record[0]}<br>"
            f"<strong>꿈 내용:</strong> {record[1]}<br>"
            f"<strong>감정:</strong> {record[2]}<br>"
            f"<strong>검색 결과:</strong><br>{record[3]}<br>"
            f"<form method='post' action='/delete'><input type='hidden' name='record_id' value='{record[0]}'><button type='submit'>삭제</button></form></p><hr>"
            for record in records
        ])
    else:
        records_html = "<p>저장된 기록이 없습니다.</p>"

    return render_template_string(f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>저장된 기록</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background: #f0f8ff;
                    margin: 0;
                    padding: 0;
                    color: #333;
                }}
                header {{
                    background-color: #00509E;
                    color: white;
                    padding: 1rem;
                    text-align: center;
                    font-size: 1.5rem;
                }}
                main {{
                    max-width: 800px;
                    margin: 2rem auto;
                    padding: 2rem;
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }}
                .record {{
                    padding: 1rem;
                    border: 1px solid #ccc;
                    border-radius: 10px;
                    background-color: #f9f9f9;
                    margin-bottom: 1rem;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                button {{
                    background-color: #FF6347;
                    color: white;
                    border: none;
                    padding: 0.7rem 1rem;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: background-color 0.3s ease;
                }}
                button:hover {{
                    background-color: #FF4500;
                }}
                a {{
                    display: inline-block;
                    margin-top: 1rem;
                    padding: 0.5rem 1rem;
                    background-color: #00509E;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                }}
                a:hover {{
                    background-color: #003f7f;
                }}
            </style>
        </head>
        <body>
            <header>
                저장된 기록
            </header>
            <main>
                <h1>📜 저장된 기록</h1>
                {records_html}
                <a href="/">홈으로</a>
            </main>
        </body>
        </html>
    ''')


@app.route('/delete', methods=['POST'])
def delete_record():
    record_id = request.form['record_id']
    cursor.execute("DELETE FROM dreams WHERE id = ?", (record_id,))
    conn.commit()
    return render_template_string('''
        <h1>기록 삭제 완료</h1>
        <p>선택한 기록이 삭제되었습니다.</p>
        <a href="/records">기록 보기로 돌아가기</a><br>
        <a href="/">홈으로</a>
    ''')

if __name__ == "__main__":
    app.run(debug=True)
