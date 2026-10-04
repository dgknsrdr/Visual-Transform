import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split,Subset
import torch.nn as nn
def to_rgb(img):
    return img.convert("RGB")


train_transform = transforms.Compose([
    transforms.Lambda(to_rgb),
    transforms.Resize(size=(224,224)),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.ColorJitter(0.2,0.2,0.2,0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5458, 0.5287, 0.5020],std=[0.3113, 0.3058, 0.3182])
])

test_transform = transforms.Compose([
    transforms.Lambda(to_rgb),
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5458, 0.5287, 0.5020],std=[0.3113, 0.3058, 0.3182])
])


train_full = datasets.Caltech101(root="data_sets",
                                 transform=train_transform,
                                 download=True
                                 )
test_full = datasets.Caltech101(root="data_sets",
                                transform=test_transform,
                                download=False)
class_names = train_full.categories
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
n_total = len(train_full)
n_test = int(n_total*0.2)
n_train = n_total-n_test

g = torch.Generator().manual_seed(42)
train_idx , test_idx = random_split(range(len(train_full)),[n_train,n_test],generator=g)

train_data = Subset(train_full,list(train_idx))
test_data = Subset(test_full,list(test_idx))


train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
test_loader= DataLoader(test_data,batch_size=32,shuffle=False)




class Patching(nn.Module):
    def __init__(self,
                 patch_size,
                 emb_num
                 ):

        super().__init__()
        self.patching = nn.Conv2d(3,emb_num,patch_size,patch_size)
        self.flatten = nn.Flatten(2,3)

    def forward(self,x):
        return (self.flatten(self.patching(x))).permute(0,2,1)


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 emb_num,
                 num_heads,
                 multi_dropout,
                 ):

        super().__init__()
        self.layer = nn.LayerNorm(normalized_shape=emb_num)
        self.mha = nn.MultiheadAttention(num_heads=num_heads,embed_dim=emb_num,dropout=multi_dropout,batch_first=True)

    def forward(self,x):
        x= self.layer(x)
        attention_out , attention_weight = self.mha(
            query=x,
            key = x,
            value = x
        )
        return attention_out

class MLPBlock(nn.Module):
    def __init__(self,
                 emb_num,
                 mlp_dropout,
                 ):
        super().__init__()
        self.layer = nn.LayerNorm(normalized_shape=emb_num)
        self.mlp = nn.Sequential(
            nn.Linear(emb_num,emb_num*4),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(emb_num*4,emb_num),
            nn.Dropout(p=mlp_dropout)
        )

    def forward(self,x):
        return self.mlp(self.layer(x))


class TransformerEncoderBlock(nn.Module):
    def __init__(self,emb_num,num_heads,mlp_dropout,multi_dropout):
        super().__init__()

        self.mha = MultiHeadAttention(emb_num,num_heads,multi_dropout)
        self.mlp = MLPBlock(emb_num,mlp_dropout)

    def forward(self, x):

        # 1. residual connection
        x = x + self.mha(x)

        # 2. residual connection
        x = x + self.mlp(x)

        return x



class VIT(nn.Module):
    def __init__(self,num_classes,
                 num_patch:int=196,
                 patch_size: int = 16,
                 emb_num: int = 768,
                 num_heads: int = 12,
                 multi_dropout: float = 0.0,
                 mlp_dropout: float = 0.1,
                 num_transform_encoder_repeat:int=12
                 ):
        super().__init__()
        self.patching = Patching(patch_size,emb_num)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_num))
        self.possition_embedding = nn.Parameter(torch.randn(1,num_patch+1,emb_num))
        self.transformer_encoder =nn.Sequential(* [TransformerEncoderBlock(emb_num,num_heads,mlp_dropout,multi_dropout) for _ in range(num_transform_encoder_repeat)])
        self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape=emb_num),
                                        nn.Linear(emb_num,num_classes))

    def forward(self,x):
        x=self.patching(x)
        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token,x),dim=1)
        x=self.possition_embedding+x
        x=self.transformer_encoder(x)
        x=self.classifier(x[:,0])
        return x


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metrics across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

vit = VIT(num_classes=len(class_names)).to(device)
optimizer = torch.optim.Adam(params=vit.parameters(),
                             lr=3e-3, # Base LR from Table 3 for ViT-* ImageNet-1k
                             betas=(0.9, 0.999), # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                             weight_decay=0.3) # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k

# Setup the loss function for multi-class classification
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model and save the training results to a dictionary
results = train(model=vit,
                       train_dataloader=train_loader,
                       test_dataloader=test_loader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=10,
                       device=device)