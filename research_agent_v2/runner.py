"""Bounded recipe runner using the official evaluator and no test split."""
from __future__ import annotations

import time
import numpy as np

from evaluate import evaluate
from research_agent.models import FM
from research_agent_v2.data import encode
from research_agent_v2.torch_models import make_model

def _run_torch(encoded, dim, fields, family, epochs, seed, deadline):
    import torch
    torch.manual_seed(seed); train,valid=encoded["train"],encoded["valid"]
    model=make_model(family,dim,len(fields)); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    x=torch.from_numpy(train["X"]).long(); y=torch.from_numpy(train["y"]).float(); rng=np.random.default_rng(seed)
    best,best_state,best_epoch,stale=-np.inf,None,0,0
    for epoch in range(1,epochs+1):
        if deadline and time.time()>=deadline: raise TimeoutError("全局时间预算已到")
        model.train()
        order=rng.permutation(len(y))
        for start in range(0,len(y),8192):
            idx=torch.from_numpy(order[start:start+8192]); pred=model(x[idx]); loss=torch.nn.functional.binary_cross_entropy_with_logits(pred,y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): scores=model(torch.from_numpy(valid["X"]).long()).numpy()
        metrics=evaluate(valid["users"],valid["y"],scores)
        if metrics["primary"]>best+1e-5:
            best,best_epoch,stale=metrics["primary"],epoch,0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=3: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): scores=model(torch.from_numpy(valid["X"]).long()).numpy()
    return model,evaluate(valid["users"],valid["y"],scores),{"best_epoch":best_epoch,"dim":dim,"fields":fields,"backend":"torch"}


def run_recipe(splits, recipe, epochs=18, seed=0, deadline=None):
    encoded, dim, fields = encode(splits, recipe["use_author_history"], recipe.get("use_history_count", False), recipe.get("use_hour_weekday", False), recipe.get("watch_bucket", False))
    train, valid = encoded["train"], encoded["valid"]
    if recipe.get("model_family") in {"deepfm","dcn"}:
        return _run_torch(encoded,dim,fields,recipe["model_family"],recipe.get("epochs",epochs),seed,deadline)
    heads = len(train["aux_names"]) + 1 if recipe["use_aux"] else 1
    model = FM(dim, k=16, lr=.001, seed=seed, heads=heads)
    rng, best_score, best_state, best_epoch, stale = np.random.default_rng(seed), -np.inf, None, 0, 0
    for epoch in range(1, epochs + 1):
        if deadline and time.time() >= deadline: raise TimeoutError("全局时间预算已到")
        order, losses = rng.permutation(len(train["y"])), []
        for start in range(0, len(order), 8192):
            idx = order[start:start + 8192]
            targets = [train["aux"][key][idx] for key in train["aux_names"]] if recipe["use_aux"] else None
            losses.append(model.step_pointwise(train["X"][idx], train["y"][idx], targets,
                                               aux_weight=recipe["aux_weight"]))
        metrics = evaluate(valid["users"], valid["y"], model.predict(valid["X"]))
        if metrics["primary"] > best_score + 1e-5:
            best_score, best_epoch, stale = metrics["primary"], epoch, 0
            best_state = {key: value.copy() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 4: break
    model.load_state_dict(best_state)
    scores = model.predict(valid["X"])
    return model, evaluate(valid["users"], valid["y"], scores), {
        "best_epoch": best_epoch, "dim": dim, "fields": fields, "last_loss": float(np.mean(losses))}
