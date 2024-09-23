import torch
from transformers import BertTokenizer

def predict_text(text, model, tokenizer, device, max_len=160):
    """
    주어진 텍스트의 감정을 예측하는 함수.
    
    Args:
    - text (str): 입력 텍스트
    - model (torch.nn.Module): 학습된 모델
    - tokenizer (BertTokenizer): BERT 토크나이저
    - device (torch.device): 학습 장치 (CPU or GPU)
    - max_len (int): 텍스트 최대 길이
    
    Returns:
    - int: 예측된 감정 레이블 (예: 1: 긍정, 0: 부정)
    """
    model = model.eval()
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)

    return prediction.item()
