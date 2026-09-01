"""Compile safe operations into reproducible GPU experiment recipes."""
def compile_recipes(operations, gpu_available=True):
    recipes = [
        {"name":"deepfm_v2_control","family":"deepfm","embedding_dim":8,"mlp_dims":[64,32],"aux_tasks":[],"aux_weight":0.0},
        {"name":"dcn_v2_control","family":"dcn","embedding_dim":8,"mlp_dims":[64,32],"aux_tasks":[],"aux_weight":0.0},
        {"name":"deepfm_gpu","family":"deepfm","embedding_dim":16 if gpu_available else 8,"mlp_dims":[128,64,32],"aux_tasks":[],"aux_weight":0.0},
    ]
    names = {item["操作"] for item in operations}
    if "din_history_attention" in names:
        recipes.append({"name":"din_primary","family":"din","embedding_dim":16,"mlp_dims":[128,64,32],"aux_tasks":[],"aux_weight":0.0})
    if "multibehavior_auxiliary" in names:
        recipes.append({"name":"din_multitask","family":"din","embedding_dim":16,"mlp_dims":[128,64,32],"aux_tasks":["click","like","hate"],"aux_weight":0.1})
    return recipes
