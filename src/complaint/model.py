from transformers import AutoTokenizer, AutoModelForSequenceClassification


def build_tokeniser(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    return tok


def build_model(model_name, label2id, id2label):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
