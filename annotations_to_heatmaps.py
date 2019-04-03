
'''
A function that returns 19 feature maps containing heatmaps corresponding to 18 keypoints (nose, neck , arm ...etc)
and background of an image containing a single person
input:
  annotations: an array of pairs of size 18, each pair is x,y coordinates of a keypoint
  
'''
def get_heatmap(annotations, desired_height, desired_width, scale=4):
    coco_parts = 19
    heatmap = np.zeros((coco_parts,
                        desired_height,
                        desired_width), dtype=np.float32)

    def put_heatmap(plane_idx, center, sigma):
        center_x, center_y = center
        _, height, width = heatmap.shape[:3]
        
        # vectorized_start
        
        th = 4.6052*(0.75)
        delta = math.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigma + 0.5))
        y0 = int(max(0, center_y - delta * sigma + 0.5))

        x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
        y1 = int(min(height - 1, center_y + delta * sigma + 0.5))

        exp_factor = 1 / 2.0 / sigma / sigma

        arr_heatmap = heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1]
        y_vec = (np.arange(y0, y1 + 1) - center_y)**2
        x_vec = (np.arange(x0, x1 + 1) - center_x)**2
        xv, yv = np.meshgrid(x_vec, y_vec)
        arr_sum = exp_factor * (xv + yv)
        arr_exp = np.exp(-arr_sum)
        arr_exp[arr_sum > th] = 0
        heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1] = np.maximum(arr_heatmap, arr_exp)

        # vectorized_end 

    for idx, point in enumerate(annotations):
        if point[0] < 0 or point[1] < 0:
            continue
        put_heatmap(idx, point, 6)

    
    heatmap[-1] = np.clip(1 - np.amax(heatmap, axis=0), 0.0, 1.0)
    ch_ = np.array([], dtype= np.float32)
    
    if scale:
      for ch in heatmap:
        ch__ = cv2.resize(ch, (desired_height // scale, desired_width // scale), interpolation=cv2.INTER_AREA)
        ch_ = np.append(ch_,ch__)
      heatmap = np.reshape(ch_, (coco_parts, desired_height // scale, desired_width // scale ))
    
    return heatmap
