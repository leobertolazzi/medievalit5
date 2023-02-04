import os 
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

# ita2dante dataset class
class Ita2Dante(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item
    def __len__(self):
        return len(self.labels['input_ids'])

# get an untrained model
def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


if __name__ == "__main__":

    # create checkpoint path
    if not os.path.exists("model"):
        os.mkdir("model")
    
    # dataset with italian and dantean text
    ita2dante_df = pd.read_csv('data/ita2dante.csv')
    ita2dante_df = ita2dante_df.applymap(str)
    train_df, test_df = train_test_split(ita2dante_df, test_size=0.05, random_state=36)

    # model name on the huggingface hub and tokenizer
    model_checkpoint = "gsarti/it5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # encode data
    train_encodings = tokenizer(train_df["italian"].to_list(), max_length=128, truncation=True)
    val_encodings = tokenizer(test_df["italian"].to_list(), max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        train_labels = tokenizer(train_df["dante"].to_list(), max_length=128, truncation=True)
        val_labels = tokenizer(test_df["dante"].to_list(), max_length=128, truncation=True)

    # Ita2Dante
    train_dataset = Ita2Dante(train_encodings, train_labels)
    val_dataset = Ita2Dante(val_encodings, val_labels)

    # model name and saving directory
    model_name = "dante-it5"
    model_dir = "model"

    # trainer arguments
    args = Seq2SeqTrainingArguments(
        model_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=200,
        learning_rate=8e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_total_limit=1,
        num_train_epochs=8,
        predict_with_generate=True,
        fp16=False,
    )

    # data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer)

    # trainer
    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # start training
    trainer.train()