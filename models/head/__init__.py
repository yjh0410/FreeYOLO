from .decoupled_head import DecoupledHead


# build detection head
def build_head(cfg, in_dim):
    head = DecoupledHead(in_dim, cfg) 

    return head