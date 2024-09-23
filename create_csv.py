import pandas as pd
import os

# 샘플 데이터 생성
data = {
    'text': [
        "I love this movie!",
        "This is the worst movie ever.",
        "I really liked the acting in this film.",
        "The plot was terrible and boring.",
        "What a fantastic experience!",
        "I hated every minute of it."
    ],
    'label': [
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
        'negative'
    ]
}

# 데이터프레임 생성
df = pd.DataFrame(data)

# 디렉터리 생성
os.makedirs('data/processed', exist_ok=True)

# CSV 파일로 저장 (쉼표로 구분)
df.to_csv('data/processed/train.csv', index=False)

print("Sample data created and saved to 'data/processed/train.csv'")
