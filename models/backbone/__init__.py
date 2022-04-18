from .darknet53 import darknet53


def build_backbone(model_name='darknet53'):
    print('==============================')
    print('Backbone: {}'.format(model_name.upper()))
    print('--pretrained: {}'.format())

    if model_name == 'darknet53':
        model, feat_dim = darknet53()

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
