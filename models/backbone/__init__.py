from .darknet import darknet53


def build_backbone(model_name='darknet53', pretrained=False):
    print('==============================')
    print('Backbone: {}'.format(model_name.upper()))
    print('--pretrained: {}'.format())

    if model_name == 'darknet53':
        model, feat_dim = darknet53(pretrained=pretrained)

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
