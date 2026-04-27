import pickle

with open("detections.pkl", "rb") as f:
    frames = pickle.load(f)

print("Frames:", len(frames))
print("Sample:", frames[0]) 
