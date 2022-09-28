import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class Embedder:
    def __init__(self, model_name):
        # self.tokenizer, self.model = build_pretrained_model(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).cuda()
        self.model.eval()

    def forward(self, node2descs):
        embs = {}
        with torch.no_grad():
            for idx, (node, desc) in tqdm(enumerate(node2descs.items())):
                inputs = self.tokenizer(desc, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = self.model(**inputs)
                # emb = outputs.pooler_output.squeeze(0).cpu().numpy()
                
                # remove fist and last token, mean pooling
                emb = torch.mean(outputs.last_hidden_state.squeeze(0)[1:-1], dim=0).cpu().numpy()
                embs[node] = emb

        return embs