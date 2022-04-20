from torch import optim


def build_optimizer(model,
                    base_lr=0.0,
                    name='sgd',
                    momentum=0.,
                    weight_decay=0.):
    print('==============================')
    print('Optimizer: {}'.format(name))
    print('--momentum: {}'.format(momentum))
    print('--weight_decay: {}'.format(weight_decay))

    if name == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                                lr=base_lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    elif name == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                                lr=base_lr,
                                weight_decay=weight_decay)
                                
    elif name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), 
                                lr=base_lr,
                                weight_decay=weight_decay)
                                
    return optimizer
