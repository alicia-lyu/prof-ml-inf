from transformers import T5ForConditionalGeneration

import time
## Define hook functions
take_time_dict = {}

def take_time_pre(layer_name,module, input):
    take_time_dict[layer_name] = time.time() 

def take_time(layer_name,module, input, output):
    take_time_dict[layer_name] =  time.time() - take_time_dict[layer_name]
    ## for TensorBoard you should use writter


# Load the T5 model
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)


print("\nAll Layers in the Model:")
for name, module in model.named_modules():
    print(f"{name}: {module}")
    print()

# # Print the encoder layers
# print("\nEncoder Layers:")
# for i, layer in enumerate(model.encoder.block):
#     print(f"Encoder Layer {i}:")
#     print(layer)

# # Print the decoder layers
# print("\nDecoder Layers:")
# for i, layer in enumerate(model.decoder.block):
#     print(f"Decoder Layer {i}:")
#     print(layer)

# # Print other components
# print("\nOther Components:")
# print("Shared Embedding Layer:")
# print(model.shared)

# print("\nFinal Layer Norm (Encoder):")
# print(model.encoder.final_layer_norm)

# print("\nFinal Layer Norm (Decoder):")
# print(model.decoder.final_layer_norm)

# print("\nLM Head:")
# print(model.lm_head)
