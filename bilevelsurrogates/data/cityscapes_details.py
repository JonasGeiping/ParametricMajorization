import numpy as np


class CityscapesSemantic:
    """
        cityscapes specifics from https://github.com/meetshah1995/pytorch-semseg/
                                        blob/master/ptsemseg/loader/cityscapes_loader.py
        and https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """
    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    n_classes = 19
    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    valid_classes = [
        7,
        8,
        11,
        12,
        13,
        17,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        31,
        32,
        33,
    ]
    class_names = [
        "unlabelled",
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic_light",
        "traffic_sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]
    class_map = dict(zip(valid_classes, range(19)))
    ignore_index = 250

    def decode_segmap(temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, CityscapesSemantic.n_classes):
            r[temp == l] = CityscapesSemantic.label_colours[l][0]
            g[temp == l] = CityscapesSemantic.label_colours[l][1]
            b[temp == l] = CityscapesSemantic.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(mask):
        # Put all void classes to zero
        for _voidc in CityscapesSemantic.void_classes:
            mask[mask == _voidc] = CityscapesSemantic.ignore_index
        for _validc in CityscapesSemantic.valid_classes:
            mask[mask == _validc] = CityscapesSemantic.class_map[_validc]
        return mask
