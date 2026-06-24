import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveScanFallback(nn.Module):
    """
    Pure PyTorch implementation of the Selective Scan SSM.
    Replicates the CUDA selective scan operations for CPU/MPS.
    """
    def __init__(self, d_inner: int, d_state: int):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

    def stable_zoh_b(self, u: torch.Tensor) -> torch.Tensor:
        """
        Computes (exp(u) - 1) / u in a numerically stable way.
        """
        # u is expected to be negative since A is initialized negative and Delta > 0.
        exp_u_minus_1 = torch.expm1(u)
        
        # Taylor series for small u to avoid division by zero
        taylor = 1.0 + 0.5 * u + (1.0 / 6.0) * (u ** 2) + (1.0 / 24.0) * (u ** 3)
        
        # Ensure we never divide by zero or tiny numbers in the division branch under backpropagation
        u_safe = torch.where(torch.abs(u) > 1e-4, u, torch.ones_like(u) * 1e-4)
        res = torch.where(torch.abs(u) > 1e-4, exp_u_minus_1 / u_safe, taylor)
        return res

    def forward(self, x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, L, D)
            delta: step size tensor of shape (B, L, D)
            A: SSM transition parameter of shape (D, N)
            B: selective control matrix of shape (B, L, N)
            C: selective measurement matrix of shape (B, L, N)
        Returns:
            y: output tensor of shape (B, L, D)
        """
        B_size, L_size, D_size = x.shape
        N_size = self.d_state
        device = x.device
        
        # A is negative. Let's ensure it is clamped or kept negative.
        A_neg = -torch.clamp(torch.abs(A), min=1e-3) # (D, N)
        
        # Initialize hidden state h: (B, D, N)
        h = torch.zeros(B_size, D_size, N_size, device=device)
        ys = []
        
        for t in range(L_size):
            # Extract time step slice
            x_t = x[:, t, :] # (B, D)
            delta_t = delta[:, t, :] # (B, D)
            B_t = B[:, t, :] # (B, N)
            C_t = C[:, t, :] # (B, N)
            
            # Discretization parameters
            # u = Delta * A -> (B, D, N)
            u = delta_t.unsqueeze(-1) * A_neg.unsqueeze(0) # (B, D, N)
            
            # A_bar = exp(Delta * A) -> (B, D, N)
            A_bar = torch.exp(u)
            
            # B_bar = (Delta * A)^-1 (exp(Delta * A) - I) * Delta * B_t
            # Let B_bar = stable_zoh_b(u) * Delta_t * B_t
            # delta_t is (B, D), B_t is (B, N). Outer product delta_t * B_t is (B, D, N)
            delta_B_t = delta_t.unsqueeze(-1) * B_t.unsqueeze(1) # (B, D, N)
            B_bar = self.stable_zoh_b(u) * delta_B_t # (B, D, N)
            
            # Update hidden state: h_t = A_bar * h_{t-1} + B_bar * x_t
            # x_t is (B, D), we expand it to (B, D, 1) to multiply with B_bar
            h = A_bar * h + B_bar * x_t.unsqueeze(-1) # (B, D, N)
            
            # Compute output: y_t = C_t * h_t -> sum over N
            # C_t is (B, N), expanded to (B, 1, N)
            y_t = torch.sum(h * C_t.unsqueeze(1), dim=-1) # (B, D)
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1) # (B, L, D)
        return y


class MambaBlock(nn.Module):
    """
    A single Mamba selective SSM layer.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # LayerNorm input normalization
        self.norm = nn.LayerNorm(self.d_model)
        
        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # 1D Convolution over sequence
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1
        )
        
        # SSM parameters
        # A parameter initialized as negative numbers for stability
        A_init = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A = nn.Parameter(A_init)
        self.A._no_weight_decay = True
        
        # Selective projections
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Discretization bias initialization
        dt_init = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_proj.bias = nn.Parameter(dt_init)
        self.dt_proj.bias._no_weight_decay = True
        
        # Selective scan implementation
        self.selective_scan = SelectiveScanFallback(self.d_inner, self.d_state)
        
        # Out projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (B, L, D)
        Returns:
            tensor of shape (B, L, D)
        """
        # Save residual
        residual = x
        
        # Normalize input (Pre-LayerNorm)
        x_norm = self.norm(x)
        
        # Project to intermediate states
        projected = self.in_proj(x_norm) # (B, L, 2 * d_inner)
        u, z = projected.chunk(2, dim=-1) # (B, L, d_inner), (B, L, d_inner)
        
        # Apply 1D convolution
        # Conv1d expects shape (B, C, L)
        u_conv = u.transpose(1, 2)
        u_conv = self.conv1d(u_conv)[:, :, :x.shape[1]] # Slice padding
        u_conv = u_conv.transpose(1, 2) # (B, L, d_inner)
        
        # Activation
        u_act = F.silu(u_conv)
        
        # Selective projection for B, C, delta
        # Projects u_act -> (B, L, d_state + d_state + d_inner)
        x_proj_out = self.x_proj(u_act)
        B, C, delta_raw = torch.split(x_proj_out, [self.d_state, self.d_state, self.d_inner], dim=-1)
        
        # Discretization step size Delta_t = Softplus(dt_proj(delta_raw))
        delta = F.softplus(self.dt_proj(delta_raw)) # (B, L, d_inner)
        delta = torch.clamp(delta, min=1e-4, max=1.0)
        
        # Selective SSM Scan
        y_ssm = self.selective_scan(u_act, delta, self.A, B, C) # (B, L, d_inner)
        
        # Gating connection: y * silu(z)
        gated = y_ssm * F.silu(z)
        
        # Output projection
        out = self.out_proj(gated) # (B, L, d_model)
        
        # Residual connection
        return out + residual


class MambaEncoder(nn.Module):
    """
    Multi-layer selective Mamba sequence encoder.
    """
    def __init__(self, num_features: int = 4, d_model: int = 64, d_state: int = 16, 
                 d_conv: int = 4, expand: int = 2, num_layers: int = 2):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        
        # Initial projection from feature size to model dimension
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Mamba layers stack
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: tensor of shape (B, L, num_features)
        Returns:
            seq_embeddings: (B, L, d_model) representation of each token
            pooled_embedding: (B, d_model) average representation of the sequence
        """
        h = self.input_projection(x) # (B, L, d_model)
        
        for layer in self.layers:
            h = layer(h)
            
        seq_embeddings = self.norm(h)
        # Pool across sequence dimension (average pooling)
        pooled_embedding = torch.mean(seq_embeddings, dim=1)
        
        return seq_embeddings, pooled_embedding
