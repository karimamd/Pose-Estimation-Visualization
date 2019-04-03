from imports import *
'''
A class encapsulating all functions used 
to utilize the cloud resources in order to 
upload and download saved models
'''

class cloud_:
  def __init__(self, username, password):
    
    self.username = username
    self.password = password
    self.pc = PyCloud(username, password)
    self.usr_enc = urllib.parse.quote(username, safe='')
    self.pass_enc = urllib.parse.quote(password, safe='')
    self.fs = opener.open_fs('pcloud://'+ self.usr_enc + ':' + self.pass_enc + '@/')

  def upload(self, dest, *files):
    self.pc.uploadfile(path= dest, files=[*files])
  
  def download(self, source, file):
    with open(file, 'wb') as f:
      self.fs.download(source, f)
  
  def exists(self, path):
    return self.fs.exists(path)
  
  def isCorrupt(self, base, ref):
    first_hash = self.fs.hash(ref, "md5")
    sec_hash = self.md5sum(base)
    return first_hash != sec_hash
      
  def md5sum(self, filename):
    md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(512 * md5.block_size), b''):
            md5.update(chunk)
    return md5.hexdigest()
  
  def delete(self, f_path):
    self.fs.remove(f_path)

'''
function downloads a model from pcloud and loads its weights into a similar blue print model passed and returns it
'''
def get_model(model, model_path, model_file, cloudp):
  
  if cloudp.exists(model_path,model_file):
      
      start_epoch=None
      best_train_loss= None
      checkpoint = None
      cloudp.download(model_path, model_file)
      print("=> loading checkpoint '{}'".format(model_file))
      checkpoint = torch.load(model_file)
      start_epoch = checkpoint['epoch']
      best_train_loss = checkpoint['best_train_loss']
      model.load_state_dict(checkpoint['state_dict'])
      print("=> loaded checkpoint '{}' (epoch {})".format(model_file,
              checkpoint['epoch']))
  else:
      print('No saved model found on cloud')
  return model  
      