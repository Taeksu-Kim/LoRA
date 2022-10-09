import loralib as lora

class LoRA_converter():
    def __init__(self, 
                 base_model,
                 tar_linear_layers={},
                 tar_embedding_layers={},
                 keep_layers={},
                 lora_r=0, 
                 lora_alpha=1, 
                 lora_dropout=0.0):
      
        self.base_model = base_model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        self.apply_lora_linear(tar_linear_layers)
        self.apply_lora_embedding(tar_embedding_layers)

        lora.mark_only_lora_as_trainable(self.base_model)

        for name, parameter in self.base_model.named_parameters():
            for tar_module in keep_layers.keys():
                if tar_module in name:
                    for tar_name in keep_layers[tar_module]:
                        if tar_name in name:
                            parameter.requires_grad = True


    # add lora layer by get_submodule
    def apply_lora_linear(self, tar_linear_layers):

        # replace submodule_key in model with module 
        module_list = self.get_module_list()
        for submodule_key in module_list:
            # load the original state dict

            for tar_module in tar_linear_layers.keys():
                # source code of loralib only replace query and value
                if submodule_key.split('.')[-1] in tar_linear_layers[tar_module]: # Linear should be replaced    
                    submodule = self.base_model.get_submodule(submodule_key)

                    # load the original state dict
                    module_state_dict = submodule.state_dict()
                
                    lora_layer = lora.Linear(
                        submodule.in_features,
                        submodule.out_features,
                        r = self.lora_r,
                        lora_alpha = self.lora_alpha,
                        lora_dropout = self.lora_dropout,
                    )

                    lora_layer.load_state_dict(module_state_dict,strict=False)
                    self.set_module(self.base_model, submodule_key, lora_layer)
                    print("Replace " + submodule_key + " with lora linear")

    # add embedding by get_module
    def apply_lora_embedding(self, tar_embedding_layers):
        module_list = self.get_module_list()

        for name in module_list:

            for tar_module in tar_embedding_layers.keys():
                chk_name = name.split('.')[-1] if '.' in name else name
                
                if chk_name in tar_embedding_layers[tar_module]:
                    submodule = self.base_model.get_submodule(name)
                    module_state_dict = submodule.state_dict()
                    
                    num_embeddings, embedding_dim = submodule.num_embeddings,submodule.embedding_dim
                    lora_layer = lora.Embedding(
                        num_embeddings,
                        embedding_dim,
                        r = self.lora_r,
                        lora_alpha = self.lora_alpha
                    )
                    
                    lora_layer.load_state_dict(module_state_dict,strict=False)
                    
                    self.set_module(self.base_model, name, lora_layer)
                    print("Replace " + name + " with lora embedding")

    def set_module(self, model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], module)
                
    def get_module_list(self):
        layer_names_dict = self.base_model.state_dict().keys()
        module_list = []

        for key in layer_names_dict:
            module_list.append('.'.join(key.split('.')[:-1]))
        
        return module_list