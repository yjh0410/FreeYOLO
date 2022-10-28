from .decoupled_head import DecoupledHead


# build detection head
def build_head(cfg):
    head = DecoupledHead(cfg) 

    return head