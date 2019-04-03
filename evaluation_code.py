'''
 this code will assume pcloud is installed and dataset is downloaded and unzipped

The function of this code is choosing multiple images to be evaluated using both the latest model and the best model
saved on pcloud and then upload those images to a folder in pcloud to qualitatively evaluate the quality of predictions


'''
from imports import *
##################################
from annotations_to_heatmaps import *
from pcloud import *
from model import *

model_path = '/models/resnet_drop/resnet_drop.pth.tar'
best_model_path = '/models/resnet_drop/best_model/resnet_drop.pth.tar'



model = TL_Model()
model.cuda()
#summary(model,(3, 224, 224))
#print(model)

best_model = TL_Model()
best_model.cuda()
#summary(best_model,(3, 224, 224))
#print(best_model)

cloudp = cloud_('zyadabozaid@hotmail.com', 'kareemzyad')

model_file = os.path.split(os.path.abspath(model_path))[-1]
model_dir = os.path.split(os.path.abspath(model_path))[0]
best_model_file = os.path.split(os.path.abspath(best_model_path))[-1]
best_model_dir = os.path.split(os.path.abspath(best_model_path))[0]



# download latest and best models on pcloud
model = get_model(model, model_path, model_file, cloudp)
best_model = get_model(best_model, best_model_path, best_model_file, cloudp)


test_idx = 8953 # index of image in dataset
ht_idx = 18 # keypoints ?
csv_path = "train_col.csv" # training csv file
data = pd.read_csv(csv_path)

# evaluate an image from the training set
img_path= data.iloc[test_idx][0]
ann = np.asarray(data.iloc[test_idx][1:])
ann =np.reshape(ann, (-1,2))
heatmap_true = get_heatmap(ann, 224, 224, 1)

loader = transforms.Compose([transforms.ToTensor()])

# img_path="sit.jpg"
image = image_loader(img_path)
model.eval()
outputs = model(image)

outputs = outputs.cpu()
outputs = outputs.detach().numpy()

# showing heatmaps individually 
for i in range(19):
  time.sleep(.1)
  print("joint ",str(i))
  x, y = np.meshgrid(np.arange(224), np.arange(224))
  plt.pcolormesh(x, y, heatmap_true[i])
  plt.colorbar() #need a colorbar to show the intensity scale
  plt.show()
  time.sleep(.1)
  plt.pcolormesh(x, y, outputs[i])
  plt.colorbar() #need a colorbar to show the intensity scale
  plt.show()

  indices = []
for i in outputs:
  indices.append(np.unravel_index(np.argmax(i), i.shape))

idx = np.asarray(indices)
# print indices
print(idx)

# display image with annotations
draw_joints(img_path,idx,1 , "annotated.jpg")

# TODO save images to files and upload them 

cloudp.upload('/evaluation','annotated.jpg')

