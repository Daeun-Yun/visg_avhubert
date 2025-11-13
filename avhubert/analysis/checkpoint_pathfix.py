import torch

ckpt_path = "/home/aristosp/ICPR26/Large/checkpoints/checkpoint_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

# Check the structure
print(ckpt.keys())
print(type(ckpt["cfg"]))
print(ckpt["cfg"].keys())

# Patch w2v_path if cfg is a dict
ckpt["cfg"]["model"]["w2v_path"] = "/home/aristosp/models/large_vox_iter5.pt"

# Save the patched checkpoint
patched_ckpt_path = ckpt_path.replace(".pt", "_fixed.pt")
torch.save(ckpt, patched_ckpt_path)
print("Patched checkpoint saved at:", patched_ckpt_path)
