import imgaug.augmenters as iaa

augmenter = iaa.Sequential([
            iaa.OneOf([
                iaa.Sometimes(
                    0.3,
                    iaa.GaussianBlur(sigma=(0, 0.5))
                ),
                iaa.Sometimes(
                    0.3,
                    iaa.Sharpen(alpha=(0.0, 0.2), lightness=(0.9, 1.1))
                )
            ]),
            iaa.SomeOf((0, 2),
            [
                iaa.Sometimes(0.4,
                    iaa.MultiplyBrightness((0.9, 1.1))
                ),
                iaa.Sometimes(0.4,
                    iaa.MultiplySaturation((0.9, 1.1))
                ),
            ]),
            iaa.Sometimes(
                0.03,
                iaa.Grayscale(alpha=(0.0, 1.0))
            )
])