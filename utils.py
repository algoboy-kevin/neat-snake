import os

def get_last_episode(folder_path):
  episodes = []
  for file in os.listdir(folder_path):
    # if file.lower().startswith('winner'):
    #   print("WINNER FOUND")
    #   return os.path.join(folder_path, file)

    if file.lower().startswith('neat-checkpoint-'):
      episodes.append(file)

  if episodes:
    filename = episodes[-1]
    file_path = os.path.join(folder_path, filename)
    return file_path
  else:
    return "No checkpoint found in the folder."
  
def does_checkpoint_exist(folder_path):
  for file in os.listdir(folder_path):
    if 'neat-checkpoint-' in file:
      return True
  return False