import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.model import EmotionDataset  # EmotionDataset을 model.py에서 임포트

def load_data(data_path):
    """
    데이터를 로드하고 전처리하는 함수.
    
    Args:
    - data_path (str): CSV 데이터 파일의 경로
    
    Returns:
    - train_df (pd.DataFrame): 학습 데이터프레임
    - val_df (pd.DataFrame): 검증 데이터프레임
    """
    # CSV 파일을 읽어 데이터프레임으로 변환
    df = pd.read_csv(data_path)

    # 간단한 전처리 (예: 결측치 제거)
    df.dropna(inplace=True)

    # 감정 레이블을 정수로 변환 (예: 긍정: 1, 부정: 0)
    df['label'] = df['label'].map({'positive': 1, 'negative': 0})

    # 학습 데이터와 검증 데이터를 8:2 비율로 나눔
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df, val_df

def preprocess_text(text):
    """
    텍스트 데이터를 전처리하는 함수.
    
    Args:
    - text (str): 원본 텍스트 데이터
    
    Returns:
    - text (str): 전처리된 텍스트 데이터
    """
    # 텍스트를 소문자로 변환
    text = text.lower()

    # 간단한 전처리 예시 (필요시 확장 가능)
    text = text.replace("\n", " ")  # 줄바꿈 제거
    text = text.strip()  # 공백 제거

    return text

def create_data_loader(df, tokenizer, max_len, batch_size):
    """
    DataLoader를 생성하는 함수.
    
    Args:
    - df (pd.DataFrame): 데이터프레임 (텍스트 및 레이블 포함)
    - tokenizer (BertTokenizer): BERT 토크나이저
    - max_len (int): 텍스트 최대 길이
    - batch_size (int): 배치 사이즈
    
    Returns:
    - DataLoader: 파이토치 DataLoader 객체
    """
    ds = EmotionDataset(
        texts=df['text'].to_numpy(),
        labels=df['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True
    )
