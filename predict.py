import sys
import time

import numpy as np
import pandas as pd
from pydantic import BaseModel

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel


class Config:
    batch_size=4
    model='UBC-NLP/MARBERTv2'
    epochs=6
    transformer_lr=1e-5 
    heads_lr=2e-5
    max_len=512
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
def print_usage():
    print("""
    ---------------------------
    Usage:
    python StanceEval.py goldFile guessFile

    goldFile: file containing gold standards;
    guessFile: file containing your prediction.

    These two files have the same format:
    ID<Tab>Target<Tab>Tweet<Tab>Stance
    Only stance labels may be different between them!
    ---------------------------
    """)


if len(sys.argv) != 2:
    sys.stderr.write("\nError: Number of parameters are incorrect!\n")
    print_usage()
    sys.exit(1)


data_pth = sys.argv[1]

data = pd.read_csv(data_pth, sep='\t')


def map_targets(target):
    if target == 'Women empowerment':
        return 'تمكين المرأة'
    if target == 'Covid Vaccine':
        return 'لقاح كوفيد 19'
    if target == 'Digital Transformation':
        return 'التحول الرقمي'


data['Target'] = data['Target'].map(map_targets)


class CustomTestDataset(Dataset):
    def __init__(self, texts, target_words, model_pth: str):
        self.texts = texts
        self.target_words = target_words
        self.tokenizer = AutoTokenizer.from_pretrained(model_pth)

    def __getitem__(self, index):
        text = self.texts[index]
        target_word = self.target_words[index]
        full_text = target_word + "[SEP]" + text 

        text_tokenized = self.tokenizer(
            full_text, max_length=512, padding="max_length", truncation=True, return_tensors='pt'
        )

        text_tokenized = {key: value.squeeze() for key, value in text_tokenized.items()}

        head_mask, no_of_sep_tokens = [], 0
        # create a head mask
        for token in text_tokenized['input_ids']:
            if no_of_sep_tokens == 1:
                head_mask.append(1) 
            else:
                head_mask.append(0) 

            if token == self.tokenizer.sep_token_id:
                no_of_sep_tokens += 1

        return text_tokenized, torch.tensor(head_mask)

    def __len__(self):
        return len(self.texts)


test_dataset = CustomTestDataset(texts=data['Tweet'].tolist(), target_words=data['Target'], model_pth=Config.model)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)



class TaskSpecificClassifier(nn.Module):
    """
    Defines a single classification task
    """

    def __init__(self, encoder_hidden_size: int, n_classes: int,
                 hidden_dropout_prob: float):
        super().__init__()
        self.n_classes = n_classes
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.pre_classifier = nn.Linear(encoder_hidden_size, encoder_hidden_size)
        self.classifier = nn.Linear(encoder_hidden_size, n_classes)
        self.relu = nn.ReLU()

    def forward(self, sentence_embedding):
        return self.classifier(self.dropout(self.relu(self.pre_classifier(sentence_embedding))))


class TaskConfig(BaseModel):
    tasks: list[dict] = []


class MultiHeadedMultiTaskModel(nn.Module):
    def __init__(
            self, model_name: str, task_config: TaskConfig,
            smoothing: float, dropout_prob: float, **kwargs
    ):
        super().__init__()
        """
        Shared encoder with multiple task specific heads
        """

        self.model = AutoModel.from_pretrained(model_name)
        self.layernorm = nn.LayerNorm(self.model.config.hidden_size)
        self.classifier_dict = nn.ModuleDict(
            {
                t["task_name"]: TaskSpecificClassifier(
                    encoder_hidden_size=self.model.config.hidden_size,
                    hidden_dropout_prob=dropout_prob,
                    n_classes=t["n_classes"],
                )
                for t in task_config.tasks
            }
        )
        self.stance_head = nn.Linear(self.model.config.hidden_size, 3)
        self._init_weights(self.stance_head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def average_pool(last_hidden_states: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
        # average pooling to get single vector representation from E5 embeddings
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(
            self,
            inputs,
            head_mask
    ):
        """
        Given a particular task and input, calculates the output logits and loss
        """
        outputs = self.model(**inputs)
        sentence_embedding = self.layernorm(self.average_pool(outputs.last_hidden_state, head_mask))

        return self.stance_head(sentence_embedding)

    
task_config = TaskConfig(
    tasks=[
        {"task_name": "hatespeech", "n_classes": 2},
        {"task_name": "sentiment", "n_classes": 3},
        {"task_name": "dialectid", "n_classes": 5},
        {"task_name": "offensive", "n_classes": 2},
        {"task_name": "sarcasm", "n_classes": 2},
    ]
)

model = MultiHeadedMultiTaskModel("UBC-NLP/MARBERTv2", task_config=task_config, smoothing=0.1, dropout_prob=0.1)
model.load_state_dict(torch.load(f"marbert-v2-multistage.bin"))
model.to(Config.device)


def predict_test(test_dataloader, model):
    model.eval()
    y_pred_list, y_true_list = [], []
    counter = 0
    start = time.time()

    with torch.no_grad():
        for X_value, head_mask in test_dataloader:
            X_value = {k: v.to(Config.device) for k, v in X_value.items()}
            head_mask = head_mask.to(Config.device)
            y_pred = model(X_value, head_mask)

            y_pred = torch.softmax(y_pred, dim=1)

            y_pred_list.append(y_pred.cpu().numpy())
            counter += 1

            if counter % 1000 == 0:
                print(f"iterations: {counter}  | time: {time.time() - start:.2f} s")

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    
    return np.array(y_pred_list)


y_pred_proba_test = predict_test(test_dataloader, model)

data['NONE_score'] = y_pred_proba_test[:, 0]
data['AGAINST_score'] = y_pred_proba_test[:, 1]
data['FAVOR_score'] = y_pred_proba_test[:, 2]

data.to_csv('scored-data.csv', index=False)

print('Data has been scored!')