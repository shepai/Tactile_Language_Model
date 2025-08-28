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
class TactileDataset(Dataset):
    def __init__(self, images, captions, tokenizer, max_len=30):
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
class TactileCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, max_len=30):
        super().__init__()
        # Image encoder: ResNet18 without classifier
        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.fc = nn.Identity()
        self.encoder = resnet

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
        self.tokenizer=AutoTokenizer.from_pretrained("t5-small")
    def train(self,X,y,epochs=100):
        """
        Pass in the X data (images) and y data (string of descriptions)
        Train the LLM
        """
        X=torch.tensor(X).reshape(X.shape[0],1,X.shape[1],X.shape[2]).float()
        #convert text to embeddings
        """text = y
        tokenized = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
        input_ids=tokenized['input_ids']
        decoded=tokenizer.decode(input_ids[0].squeeze(), skip_special_tokens=True)
        y=decoded"""
        #train model
        self.model = TactileCaptioningModel(vocab_size=self.tokenizer.vocab_size)
        device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        dataset = TactileDataset(X, y, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        for epoch in range(epochs): #train loop
            self.model.train()
            total_loss = 0
            for batch in dataloader:
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                outputs = self.model(images, input_ids[:, :-1])
                loss = loss_fn(outputs.reshape(-1, self.tokenizer.vocab_size), input_ids[:, 1:].reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")
    def generate_caption(self, image, max_len=30, device='cpu'):
        self.model.eval()
        image = torch.tensor(image).to(device)

        with torch.no_grad():
            # Step 1: Encode image
            image_features = self.model.encoder(image)  # (1, 512)
            image_features = self.model.img_proj(image_features).unsqueeze(1).transpose(0, 1)  # (1, 1, embed_dim)

            # Step 2: Start decoding
            input_ids = torch.tensor([[self.tokenizer.pad_token_id]], device=device)  # (1, 1)

            for _ in range(max_len):
                tgt = self.model.embedding(input_ids)  # (1, T, E)
                tgt = tgt.transpose(0, 1)         # (T, 1, E)

                output = self.model.transformer_decoder(tgt, image_features)  # (T, 1, E)
                last_token_logits = self.model.output_layer(output[-1])       # (1, vocab_size)
                next_token_id = torch.argmax(last_token_logits, dim=-1)  # (1,)

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

    def load(self, filename, vocab_size):
        # Create model instance with correct vocab size
        self.model = TactileCaptioningModel(vocab_size=vocab_size)
        device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        checkpoint = torch.load(filename, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {filename}")

class Decisions:
    def __init__(self,model="mistral"):
        self.MODEL=model
    def chat(self,reading):
        usermessage="Act like you are a quadruped robot with tactile sensors. Simply tell me how would you adjust your gait based on the tactile sensor reading:"+str(reading)+". Give your answer in the format where you pick one of the functions as an option 'speed action: [slowSpeed(), maintainSpeed(), increaseSpeed()] \nleg spread action: [widenLegStride(), maintainLegSTride(), increaseLegStride()] \nbody centre action [lowerBody(), maintainBody(), IncreaseBody()]'"
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
    X = np.load("/mnt/data0/drs25/data/optical-tactile-dataset-for-textures/texture-tactip/X_data_15.npz")['arr_0'].astype(np.uint8)
    y = np.load("/mnt/data0/drs25/data/optical-tactile-dataset-for-textures/texture-tactip/y_data_15.npz")['arr_0'].astype(np.uint8)
    X=X[:,0:7,:,:]
    X=X.reshape((len(X),X.shape[1]*X.shape[2],X.shape[3]))
    print(X.shape)
    def reshape(X,percent):
        w=int(X.shape[1]*percent)
        h=int(X.shape[2]*percent)
        new_array=np.zeros((X.shape[0],w,h),dtype=np.uint8)
        for i in range(X.shape[0]):
            new_array[i] = cv2.resize(X[i],(h,w),interpolation=cv2.INTER_AREA)
        return new_array
    X=reshape(X,0.5)
    keys=['Carpet', 'LacedMatt', 'wool', 'Cork', 'Felt', 'LongCarpet', 'cotton', 'Plastic', 'Flat', 'Ffoam', 'Gfoam', 'bubble', 'Efoam', 'jeans', 'Leather']
    material_descriptions = {
        "carpet": "Dense, woven fibers, soft yet coarse, typically synthetic or wool blend.",
        "lacedmatt": "Light, airy mesh structure with interwoven hard lace patterns; bumpy and flexible.",
        "wool": "Natural fiber, soft, warm, and slightly scratchy; high friction texture.",
        "cork": "Lightweight, firm but compressible, slightly rough with granular texture.",
        "felt": "Compressed fabric, soft and smooth surface, uniform texture with slight give.",
        "longcarpet": "High-pile carpet with long fibers, soft and plush, deep texture.",
        "cotton": "Smooth and soft woven fabric, breathable with moderate friction.",
        "plastic": "Hard, smooth surface with low friction; can vary from rigid to flexible.",
        "flat": "Smooth and even surface, minimal texture; likely hard or semi-soft material.",
        "ffoam": "Soft with slight springiness, absorbs pressure well.",
        "gfoam": "Grainy foam, slightly rougher texture, spongy and compressible.",
        "bubble": "Bubble wrap or bubbled plastic, soft with raised circular nodes, very bumpy.",
        "efoam": "Soft with slight springiness, absorbs pressure well.",
        "jeans": "Sturdy cotton denim, rough woven texture, moderate friction.",
        "leather": "Smooth and durable natural material, slightly soft with subtle grain."
    }
    new_y=[]
    for i in range(len(y)):
        label=keys[int(y[i])].lower()
        new_y.append(material_descriptions[label])

    #get model
    tlm = TLM()
    tlm.train(X,new_y)
    tlm.save("/its/home/drs25/Tactile_Language_Model/data/trainedModel")
    print("generated:",tlm.generate_caption(X[0][0]),tlm.tokenizer)