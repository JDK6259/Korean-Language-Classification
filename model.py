from transformers import AutoModelForSequenceClassification

def ret_model(dataset):

    model = AutoModelForSequenceClassification.from_pretrained(
        "kykim/bert-kor-base", 
        num_labels = dataset['label'].nunique(), 
        output_attentions = False, 
        output_hidden_states = False, 
    )

    return model