import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class STAM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_frames,
        num_classes,
        time_depth,
        time_heads,
        time_mlp_dim,
        time_dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.
    ):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # num_patches = (image_size // patch_size) ** 2
        # patch_dim = 3 * patch_size ** 2

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b f c (h p1) (w p2) -> b f (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        #     nn.Linear(patch_dim, dim),
        # )
        self.num_frames = num_frames
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames+1, dim))
        # self.space_cls_token = nn.Parameter(torch.randn(1, dim))
        self.time_cls_token = nn.Parameter(torch.randn(1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.time_transformer = Transformer(dim, time_depth, time_heads, time_dim_head, time_mlp_dim, dropout)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, video):

        # positional embedding
        x = video
        b, f, n = x.shape
        time_cls_tokens = repeat(self.time_cls_token, 'n d -> b n d', b=b)
        x = torch.cat((time_cls_tokens, x), dim=-2)
        x += self.pos_embedding[:, :(self.num_frames + 1)]
#         print('pos_embedding:',self.pos_embedding[:, :, :(self.num_frames + 1)].shape)
        x = self.dropout(x)

        # time attention

        x = self.time_transformer(x)

        # final mlp

        return self.mlp_head(x[:, 0])

if __name__ == "__main__":
    model = STAM(
        dim=1024,
        num_frames=50,  # number of image frames, selected out of video
        time_depth=12,  # depth of time transformer (in paper, it was shallower, 6)
        time_heads=8,  # heads of time transformer
        time_mlp_dim=2048,  # feedforward hidden dimension of time transformer
        num_classes=100,  # number of output classes
        time_dim_head=64,  # time transformer head dimension
        dropout=0.,  # dropout
        emb_dropout=0.  # embedding dropout
    )

    frames = torch.randn(2, 50, 1024)  # (batch x frames x channels x height x width)
    pred = model(frames)  # (2, 100)
    print(pred.shape)