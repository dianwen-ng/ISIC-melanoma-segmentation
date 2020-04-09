def compile_model(model, num_classes, metrics, loss, lr):
    from tensorflow.keras.losses import binary_crossentropy
    from tensorflow.keras.losses import categorical_crossentropy

    from tensorflow.keras.metrics import binary_accuracy
    from tensorflow.keras.metrics import categorical_accuracy

    from tensorflow.keras.optimizers import Adam

    from metrics import dice_coeff
    from metrics import jaccard_index
    from metrics import class_jaccard_index
    from metrics import pixelwise_precision
    from metrics import pixelwise_sensitivity
    from metrics import pixelwise_specificity
    from metrics import pixelwise_recall

    if isinstance(loss, str):
        if loss in {'ce', 'crossentropy'}:
            if num_classes == 1:
                loss = binary_crossentropy
            else:
                loss = categorical_crossentropy
        else:
            raise ValueError('unknown loss %s' % loss)

    if isinstance(metrics, str):
        metrics = [metrics, ]

    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'acc':
            metrics[i] = binary_accuracy if num_classes == 1 else categorical_accuracy
        elif metric == 'jaccard_index':
            metrics[i] = jaccard_index(num_classes)
        elif metric == 'jaccard_index0':
            metrics[i] = class_jaccard_index(0)
        elif metric == 'jaccard_index1':
            metrics[i] = class_jaccard_index(1)
        elif metric == 'jaccard_index2':
            metrics[i] = class_jaccard_index(2)
        elif metric == 'jaccard_index3':
            metrics[i] = class_jaccard_index(3)
        elif metric == 'jaccard_index4':
            metrics[i] = class_jaccard_index(4)
        elif metric == 'jaccard_index5':
            metrics[i] = class_jaccard_index(5)
        elif metric == 'dice_coeff':
            metrics[i] = dice_coeff(num_classes)
        elif metric == 'pixelwise_precision':
            metrics[i] = pixelwise_precision(num_classes)
        elif metric == 'pixelwise_sensitivity':
            metrics[i] = pixelwise_sensitivity(num_classes)
        elif metric == 'pixelwise_specificity':
            metrics[i] = pixelwise_specificity(num_classes)
        elif metric == 'pixelwise_recall':
            metrics[i] = pixelwise_recall(num_classes)
        else:
            raise ValueError('metric %s not recognized' % metric)

    model.compile(optimizer=Adam(lr=lr),
                  loss=loss,
                  metrics=metrics)


