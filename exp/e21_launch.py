"""Kick off the deployed RULER eval; results stream to the flashkv-out volume."""
import modal
f = modal.Function.from_name("flashkv-ruler", "run_eval")
call = f.spawn("Qwen/Qwen2.5-7B-Instruct", 32768, [0.005, 0.0156],
               [None, 0.0, 0.1, 0.3, 1.0], [0.15, 0.35, 0.55, 0.75, 0.9], 6, 64, 40)
print("spawned", call.object_id)
