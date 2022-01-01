

class NNModel():  # can be a diffrent name
    def __init__(self):
        pass
    
    def __freezeLayers(self, selectedModel):
        
        for param in selectedModel.parameters():
            param.requires_grad = False
    
    def __unfreezeLayer(self, selectedModel, layerNumber):
        for param in selectedModel.features[layerNumber].parameters():
            param.requires_grad = True
