import numpy as np
from datasets import load_metric


class Metrics:
    def __init__(self, processor):
        self.cer_metric = load_metric("cer")
        self.processor = processor

    def compute_metrics(self, pred):
        label_ids = pred.label_ids
        pred_ids = pred.predictions
        print(label_ids.shape, pred_ids.shape)

        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = np.array(["".join(text.split()) for text in pred_str])
        label_str = np.array(["".join(text.split()) for text in label_str])

        results = {}
        try:
            results["cer"] = self.cer_metric.compute(predictions=pred_str, references=label_str)
        except Exception as e:
            print(e)
            print(pred_str)
            print(label_str)
            results["cer"] = 0
        results["accuracy"] = (pred_str == label_str).mean()

        return results
