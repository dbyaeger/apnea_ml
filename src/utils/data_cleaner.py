from pathlib import Path
import pickle

path = Path()
data_path = path.joinpath("..", "data", "training_data", "processed")
    
  
count = 0
del_files = []
for f in data_path.iterdir():
    if f.stem == "data_partition":
        continue
    with f.open('rb') as f_in:
        data = pickle.load(f_in)
        for c in data['signals']:
            if data['signals'][c].reshape(-1, 100).shape[0] < 30:
                print(f)
                count += 1
                del_files.append(f)
                break
print(count)

for f in del_files:
    f.unlink()
    
    