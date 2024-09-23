import torch
from transformers import BertTokenizer
from src.data_loader import load_data, preprocess_text, create_data_loader
from src.model import EmotionClassifier, train_model
from src.inference import predict_text
from src.evaluation import evaluate_model

def main():
    # 파라미터 설정
    MAX_LEN = 160
    BATCH_SIZE = 16
    EPOCHS = 4
    LEARNING_RATE = 2e-5

    # 장치 설정 (GPU 사용 가능시 GPU 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드 및 전처리
    train_df, val_df = load_data("data/processed/train.csv")
    train_df['text'] = train_df['text'].apply(preprocess_text)
    val_df['text'] = val_df['text'].apply(preprocess_text)

    # 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # DataLoader 생성
    train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(val_df, tokenizer, MAX_LEN, BATCH_SIZE)

    # 모델 초기화
    model = EmotionClassifier(n_classes=2)
    model = model.to(device)

    # 옵티마이저 및 스케줄러 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 손실 함수 설정
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # 모델 학습
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        train_acc, train_loss = train_model(model, train_data_loader, loss_fn, optimizer, device, None, len(train_df))
        print(f'Train loss: {train_loss} accuracy: {train_acc}')

    # 모델 평가
    evaluate_model(model, val_data_loader, device)

    # 예측 테스트
    sample_text = "I love this movie!"
    prediction = predict_text(sample_text, model, tokenizer, device)
    print(f'Text: {sample_text}, Prediction: {prediction}')

if __name__ == "__main__":
    main()
