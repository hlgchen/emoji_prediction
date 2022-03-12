# currently this model has not been used (previously in embert.py)
class BigSembert(nn.Module):
    def __init__(self):
        super(BigSembert, self).__init__()
        self.emoji_embeddings = nn.Parameter(
            get_emoji_fixed_embedding(image=True, bert=False, wordvector=False),
            requires_grad=False,
        )

        sentence_model_name = "all-distilroberta-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"sentence-transformers/{sentence_model_name}"
        )
        self.model = AutoModel.from_pretrained(
            f"sentence-transformers/{sentence_model_name}"
        )
        self.sentence_embedding_size = 768

        description_model_name = "all-MiniLM-L6-v2"
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            f"sentence-transformers/{description_model_name}"
        )
        self.description_model = AutoModel.from_pretrained(
            f"sentence-transformers/{description_model_name}"
        )
        self.description_embedding_size = 384

        descriptions = get_emoji_descriptions()
        self.dtoken = self.description_tokenizer(
            descriptions, return_tensors="pt", truncation=True, padding=True
        )

        self.emoji_embedding_size = (
            self.emoji_embeddings.size(1) + self.description_embedding_size
        )
        self.linear1 = nn.Linear(self.sentence_embedding_size, 500)
        self.linear2 = nn.Linear(self.emoji_embedding_size, 500)

    def partial_forward(self, sentence_ls, model, batch_size):
        if isinstance(sentence_ls, list):
            encoded_input = self.tokenizer(
                sentence_ls, return_tensors="pt", truncation=True, padding=True
            )
        else:
            encoded_input = sentence_ls

        encoded_input["input_ids"] = encoded_input["input_ids"].to(device)
        encoded_input["attention_mask"] = encoded_input["attention_mask"].to(device)

        input_id_ls = torch.split(encoded_input["input_ids"], batch_size)
        attention_mask_ls = torch.split(encoded_input["attention_mask"], batch_size)
        model_output_ls = []
        for input_ids, attention_mask in zip(input_id_ls, attention_mask_ls):
            temp = model(input_ids=input_ids, attention_mask=attention_mask)
            model_output_ls.append(temp[0])
        text_model_output = torch.cat(model_output_ls, dim=0)

        embeddings = mean_pooling(text_model_output, encoded_input["attention_mask"])
        sentence_embeddings = F.normalize(embeddings, p=2, dim=1)
        return sentence_embeddings

    def forward(self, sentence_ls, emoji_ids):
        batch_size = len(sentence_ls)

        # handle twitter text
        sentences_embeddings = self.partial_forward(sentence_ls, self.model, batch_size)

        # handle emoji embedding
        dtoken_input_ids = self.dtoken["input_ids"][emoji_ids]
        dtoken_attention_mask = self.dtoken["attention_mask"][emoji_ids]
        description_tokens = {
            "input_ids": dtoken_input_ids,
            "attention_mask": dtoken_attention_mask,
        }
        description_embeddings = self.partial_forward(
            description_tokens, self.description_model, batch_size
        )
        img_embedding = self.emoji_embeddings[emoji_ids]
        emoji_embeddings = torch.cat([img_embedding, description_embeddings], dim=1)

        # combine the two
        X_1 = sentences_embeddings.repeat_interleave(len(emoji_ids), dim=0)
        X_2 = emoji_embeddings.repeat(len(sentence_ls), 1)

        X_1 = self.linear1(X_1)
        X_2 = self.linear2(X_2)
        out = (X_1 * X_2).sum(dim=1).view(-1, len(emoji_ids))
        out = F.softmax(out, dim=1)

        return out


# code not used as of now (part module of bigsembert, used to be in ee_model)
class DescriptionSembert(nn.Module):
    def __init__(self, pretrained_path=None):
        super(DescriptionSembert, self).__init__()

        model_name = "all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"sentence-transformers/{model_name}"
        )
        self.model = AutoModel.from_pretrained(f"sentence-transformers/{model_name}")
        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path, map_location=device))

    def forward(self, description_ls):
        encoded_input = self.tokenizer(
            description_ls,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        )

        input_ids = encoded_input["input_ids"].to(device)
        attention_mask = encoded_input["attention_mask"].to(device)
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = mean_pooling(model_output, attention_mask)
        sentence_embeddings = F.normalize(embeddings, p=2, dim=1)

        return sentence_embeddings
