"""
TLM Tactile language model
Prototype model attempt to combine tactile sensor readings to generalise better with describing features of the physical environment. 
We hope this steps towards more general tactile sensing applications.

Code written by Dexter R. Shepherd
PhD student at the University of Sussex

"""

import numpy as np 
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torch
from ollama import chat
import requests
import re
class TactileDataset(Dataset):
    def __init__(self, images, captions, tokenizer, max_len=300):
        self.images = images
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.captions = captions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        caption = self.captions[idx]

        tokens = self.tokenizer(caption, max_length=self.max_len,
                                padding="max_length", truncation=True,
                                return_tensors="pt")
        
        return {
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0)
        }
class SimpleTactileCNN(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),  # (B, 32, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(2),                           # (B, 32, H/4, W/4)

            nn.Conv2d(32, 64, 3, stride=1, padding=1), # (B, 64, H/4, W/4)
            nn.ReLU(),
            nn.MaxPool2d(2),                           # (B, 64, H/8, W/8)

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))               # (B, 128, 1, 1)
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)          # (B, out_dim)
class TactileCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, max_len=30):
        super().__init__()
        # Image encoder: ResNet18 without classifier
        self.encoder = SimpleTactileCNN(out_dim=512)

        # Project image features
        self.img_proj = nn.Linear(512, embed_dim)

        # Text decoder: Embedding + Transformer Decoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=4)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, images, input_ids):
        batch_size = images.size(0)
        image_features = self.encoder(images)  # (B, 512)
        image_features = self.img_proj(image_features)  # (B, E)
        image_features = image_features.unsqueeze(1)  # (B, 1, E)

        tgt = self.embedding(input_ids)  # (B, T, E)
        tgt = tgt.transpose(0, 1)  # (T, B, E)
        memory = image_features.transpose(0, 1)  # (1, B, E)

        out = self.transformer_decoder(tgt, memory)  # (T, B, E)
        out = out.transpose(0, 1)  # (B, T, E)

        return self.output_layer(out)
class TLM:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
        special_tokens_dict = {
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "pad_token": "<PAD>"
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)

        # Recreate the model with updated vocab size
        self.vocab_size = len(self.tokenizer)
        self.model = TactileCaptioningModel(vocab_size=self.vocab_size)
        #self.model = TactileCaptioningModel(vocab_size=self.tokenizer.vocab_size)
    def train(self, X, y, epochs=100, save="", lr=1e-4):
        """
        Pass in the X data (images) and y data (string of descriptions)
        Train the captioning model
        """
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32)
        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(
            label_smoothing=0.1,
            ignore_index=self.tokenizer.pad_token_id
        )

        dataset = TactileDataset(X, y, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        print("Tokenizer size:", len(self.tokenizer))
        print("Embedding size:", self.model.embedding.num_embeddings)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in dataloader:
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                outputs = self.model(images, input_ids[:, :-1])
                
                loss = loss_fn(
                    outputs.reshape(-1, self.vocab_size),
                    input_ids[:, 1:].reshape(-1)
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

            if save:
                self.save(f"{save}")

    def generate_caption(self, image, max_len=300, device='cuda'):
        self.model.to(device)
        self.model.eval()
        image=image.permute(0,3,1,2)
        image = torch.tensor(image).to(device).float()

        with torch.no_grad():
            # Step 1: Encode image
            image_features = self.model.encoder(image)  # (1, 512)
            image_features = self.model.img_proj(image_features).unsqueeze(1).transpose(0, 1)  # (1, 1, embed_dim)

            # Step 2: Start decoding
            first_token_id = self.tokenizer.convert_tokens_to_ids('{')
            input_ids = torch.tensor([[first_token_id]], device=device)  # (1, 1)

            for _ in range(max_len):
                tgt = self.model.embedding(input_ids)  # (1, T, E)
                tgt = tgt.transpose(0, 1)         # (T, 1, E)

                output = self.model.transformer_decoder(tgt, image_features)  # (T, 1, E)
                last_token_logits = self.model.output_layer(output[-1])       # (1, vocab_size)
                probs= torch.softmax(last_token_logits, dim=-1)  # (1,)
                k = 50
                topk_probs, topk_indices = torch.topk(probs, k, dim=-1)  # (1, k)
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
                sample_idx = torch.multinomial(topk_probs, num_samples=1)
                next_token_id = topk_indices.gather(-1, sample_idx) 
                next_token_id = next_token_id.squeeze(1).to(device)
                next_token_id = torch.argmax(last_token_logits, dim=-1)
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

                # Stop if EOS
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

            decoded = self.tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
            return decoded
    def generate_caption_nongreedy(self, image, max_len=300, device='cuda'):
        self.model.to(device)
        self.model.eval()
        image=image.permute(0,3,1,2)
        image = torch.tensor(image).to(device).float()

        with torch.no_grad():
            # Step 1: Encode image
            image_features = self.model.encoder(image)  # (1, 512)
            image_features = self.model.img_proj(image_features).unsqueeze(1).transpose(0, 1)  # (1, 1, embed_dim)

            # Step 2: Start decoding
            first_token_id = self.tokenizer.convert_tokens_to_ids('{')
            input_ids = torch.tensor([[first_token_id]], device=device)  # (1, 1)

            for _ in range(max_len):
                tgt = self.model.embedding(input_ids)  # (1, T, E)
                tgt = tgt.transpose(0, 1)         # (T, 1, E)

                output = self.model.transformer_decoder(tgt, image_features)  # (T, 1, E)
                last_token_logits = self.model.output_layer(output[-1])       # (1, vocab_size)
                probs = torch.softmax(last_token_logits, dim=-1)          # (1, vocab_size)
                k = 50
                topk_probs, topk_indices = torch.topk(probs, k, dim=-1)   # pick top-k tokens
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
                sample_idx = torch.multinomial(topk_probs, num_samples=1) # sample one token
                next_token_id = topk_indices.gather(-1, sample_idx).squeeze(1).to(device)
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

                # Stop if EOS
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

            decoded = self.tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
            return decoded
    def save(self, filename):
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        # Create model instance with correct vocab size
        self.model = TactileCaptioningModel(vocab_size=self.vocab_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        checkpoint = torch.load(filename, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {filename}")

class Decisions:
    def __init__(self,model="mistral"):
        self.MODEL=model
        self.usermessage="Act like you are a quadruped robot with tactile sensors. Simply tell me how would you adjust your gait based on the tactile sensor reading: READING. Give your answer in the format where you pick one of the functions as an option 'speed action: [slowSpeed(), maintainSpeed(), increaseSpeed()] \nleg spread action: [decreaseLegStride(), maintainLegSTride(), increaseLegStride()] \nbody centre action [lowerBody(), maintainBody(), IncreaseBody()]'"
        self.conceptusermessage="Act like you are a quadruped robot with tactile sensors. Simply tell me how would you adjust your gait based on the tactile sensor reading: READING. Give your answer in the format where you pick one of the functions as an option 'speed action: ... \nleg spread action: ... \nbody centre action ...'"

    def chat(self,reading):
        usermessage=self.usermessage.replace("READING",reading)
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.MODEL,
                "prompt": usermessage,
                "stream": False  # change to True for word-by-word streaming
            }
        )
        reply = ""
        if response.ok:
            reply = response.json()["response"]
        else:
            print("Error communicating with Ollama:", response.text)
        return reply



if __name__=="__main__":
    import cv2
    X=torch.rand(100,100,100,1)
    y=["{json\n\"test\":\"test\"}" for i in range(100)]

    #get model
    tlm = TLM()
    #tlm.train(X,y,epochs=200)
    tlm.load("/its/home/drs25/Tactile_Language_Model/data/models/test")
    print("generated:",tlm.generate_caption(X[0:1]))