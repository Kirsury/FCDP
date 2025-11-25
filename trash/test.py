for name, param in self.encoder.named_parameters():
    if name.startswith("encoder.layer."):
        layer_num = int(name.split(".")[2])
        if layer_num < freeze_layers:
            param.requires_grad = False