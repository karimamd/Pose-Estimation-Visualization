from imports import *

def draw_joints(img_path, ann, scale, name= "annotated.jpg"):
  image = cv2.imread(img_path)
  if scale:
    if scale != 1:
      image = cv2.resize(image, (image.shape[0] // scale, image.shape[1] // scale), interpolation=cv2.INTER_AREA)
  image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  kp = ann
  if scale:
    kp = kp // scale
  for i in kp:
    if i[0] not in range(1,image.shape[0]) or i[1] not in range(1,image.shape[1]):
      continue
    cv2.circle(image, tuple(i), 1, (0,255,0), -1)
  #plt.imshow(image)
  #plt.xticks([]), plt.yticks([])
  #plt.show()
  scipy.misc.imsave(name, image)