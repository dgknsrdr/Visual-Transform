# Vision Transformer (ViT) from Scratch with PyTorch

Bu proje, PyTorch kullanarak Vision Transformer (ViT) mimarisini sıfırdan kurup
Caltech101 veri seti üzerinde çok sınıflı görüntü sınıflandırma yapmak için hazırlanmıştır.

## Özellikler

- Caltech101 veri setini `torchvision.datasets.Caltech101` ile indirir ve yükler.
- Görselleri 224x224 boyutuna getirir.
- Eğitim sırasında horizontal flip ve color jitter augmentation uygular.
- Görüntüyü Conv2d kullanarak patch'lere ayırır.
- Class token ve positional embedding kullanır.
- Multi-Head Self-Attention ve MLP bloklarından oluşan Transformer Encoder yapısı içerir.
- Residual connection ve LayerNorm kullanır.
- Son class token üzerinden sınıflandırma yapar.
- Train/test accuracy ve loss değerlerini epoch bazında hesaplar.

## Model Yapısı

Varsayılan ViT ayarları:

- Image size: 224x224
- Patch size: 16x16
- Patch sayısı: 196
- Embedding dimension: 768
- Attention head sayısı: 12
- Transformer Encoder blok sayısı: 12
- MLP hidden dimension: 4 x embedding dimension
- MLP dropout: 0.1

Patch embedding işlemi şu yapı ile yapılır:

```python
nn.Conv2d(
    in_channels=3,
    out_channels=emb_num,
    kernel_size=patch_size,
    stride=patch_size
)
```

## Kurulum

Repository'yi klonladıktan sonra sanal ortam oluşturabilirsin:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Gerekli paketleri yükle:

```bash
pip install -r requirements.txt
```

> GPU kullanmak istiyorsan PyTorch'u ekran kartına ve işletim sistemine uygun
> resmi PyTorch kurulum komutuyla kurman gerekebilir.

## Çalıştırma

Ana Python dosyanı çalıştır:

```bash
python main.py
```

Kod ilk çalıştırmada Caltech101 veri setini `data_sets/` klasörüne indirebilir.

## Eğitim

Projede varsayılan olarak:

```python
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    params=vit.parameters(),
    lr=3e-3,
    betas=(0.9, 0.999),
    weight_decay=0.3
)
```

kullanılmaktadır.

Varsayılan eğitim süresi:

```python
epochs = 10
```

CUDA kullanılabiliyorsa otomatik olarak GPU seçilir:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Proje Yapısı

Örnek:

```text
project/
│
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
└── data_sets/        # Git'e eklenmez
```

## Not

Bu proje eğitim ve öğrenme amacıyla Vision Transformer mimarisinin temel
bileşenlerini manuel olarak uygulamaktadır.
