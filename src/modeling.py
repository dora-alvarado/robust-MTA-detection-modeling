import numpy as npfrom scipy import interpolatefrom scipy.spatial.distance import directed_hausdorff, cdistimport cv2from src.preprocessing import normalized_distance_transform, keep_largest_componentimport matplotlib.pyplot as pltfrom src.multiline_detector import im_segdef order_points(pts):    pts_aux = sorted(pts, key=lambda k: [k[1], k[0]])    pts_aux = [np.asarray(pt, dtype=np.int) for pt in pts_aux]    points= np.asarray(pts_aux, dtype=np.int)    return pointsdef residual_error_splines(points, model):    x = points[:,0]    ypred = interpolate.splev(x, model, der=0)    ytrue = points[:,1]    d = ((ypred-ytrue)**2)**0.5    # if any numerical error    d[np.isnan(d)] = np.inf    return np.asarray(d, dtype=np.float)def get_splines(points, k, s=0):    x = points[:,0]    y = points[:,1]    tck = interpolate.splrep(x, y, k=k, s=s)    return tckdef RANSAC_estimate_splines(points, n, k, epsilon=15., N=5000,  p=0.95, weights=None, flag_print=0, th_img=None, draw_models=False):    row_min = np.min(points[:,0])    row_max = np.max(points[:, 0])    partition = (row_max-row_min)/3.    cut1 = np.argwhere(points[:,0]<=(partition + row_min))    cut2 = np.argwhere(((partition + row_min)<points[:,0]) & (points[:,0]<=(2*partition+row_min)))    cut3 = np.argwhere(2*partition+row_min<points[:,0])    draw_im = np.zeros(th_img.shape)    size = len(points)    mask = np.ones((len(points),), dtype=np.bool)    max_n_inliers = 0    n_weights = weights/np.sum(weights)    best_error = np.inf    best_model = None    datapoints = None    knots = None    for i in range(N):        n_kp = 0        idx = []        while n_kp<n:            kp1 = np.random.choice(len(cut1), 1, p=None, replace=False)            kp2 = np.random.choice(len(cut2), 1, p=None, replace=False)            kp3 = np.random.choice(len(cut3), 1, p=None, replace=False)            n_kp+=3            idx.append(cut1[kp1])            idx.append(cut2[kp2])            idx.append(cut3[kp3])        # select n random points        np.random.shuffle(idx)        idx = np.asarray(idx[:n], dtype=np.int).ravel()##        # sort selected points to calculate splines        selected = points[idx]        idx_sort = np.argsort(selected[:,0])        sort_selected = selected[idx_sort]        # calculate splines        model = get_splines(sort_selected, k)        errors = residual_error_splines(points, model)        inliers = errors < epsilon        n_inliers = np.sum(inliers*n_weights)*size        # calculate total model's error        total_error = np.sum(errors*weights)        # adaptive RANSAC        rho = n_inliers / size        den = 1 - rho ** n#(k*n-1)        if np.abs(1.0 - den) > 1e-8 and den>0: #            N = np.log(1 - p) / np.log(den)        # save the best model: more weigthed inliers and smaller total error        if n_inliers > max_n_inliers and total_error<best_error:            max_n_inliers = n_inliers            mask = np.copy(inliers)            best_error = total_error            best_model = np.copy(model)            # we also save the keypoints (the points we used to calculate the splines)            datapoints = sort_selected            knots = np.zeros((best_model[0].shape[0], 2))            knots[:, 0] = best_model[0]            knots[:, 1] = np.round(interpolate.splev(best_model[0], best_model))            if flag_print>1:                print('[%d] inliers: %d, w_inliers: %d, error: %.4f' % (i, np.sum(inliers),max_n_inliers, best_error))        if draw_models:            im_mta = (draw_mta_model(th_img, model, keypoints=datapoints, lineThickness=2, vmin=0, vmax=th_img.shape[0]-1, truncate=False) / 255.)            draw_im =np.max([im_mta*n_inliers, draw_im], axis=0)# im_mta[im_mta>0]*n_inliers#        if N <= i and n_inliers >= n:            break    if flag_print:        print('%d: %d / %d, error: %.4f' % (i, max_n_inliers, len(points), best_error))    if draw_models:        im_mta = (draw_mta_model(th_img, best_model, keypoints=datapoints, lineThickness=2, vmin=0, vmax=th_img.shape[0] - 1,                                 truncate=False) / 255.)        draw_im = np.max([im_mta * max_n_inliers, draw_im], axis=0)  # im_mta[im_mta>0]*n_inliers#        return best_model, mask, datapoints, knots, draw_im    return best_model, mask, datapoints, knotsdef get_side(keypoints, img_width):    cols_mean = np.unique(keypoints[:,1]).mean()    # Left side    if cols_mean<=img_width/2.:        return 1    # Right side    return 0def eval_splines(model, keypoints, points, img_size):    # get ground-truth points    y_true = points[:, 1]    x = points[:, 0]    vmin = np.min(keypoints[:, 0])  # points[:, 0])    vmax = np.max(keypoints[:, 0])  # points[:, 0])    x_pred = np.asarray(np.arange(vmin, vmax + 1), dtype=int)    y_pred = np.asarray(interpolate.splev(x_pred, model, der=0), dtype=int)    # get predictions    # if predicted value outside of the image, clip it    y_pred = np.clip(y_pred, 0, img_size[1])    if keypoints is not None:        side = get_side(keypoints, img_size[1])        if side:            idx = y_pred <= img_size[1] / 2.        else:            idx = y_pred > img_size[1] / 2.        x_pred = x_pred[idx]        y_pred = y_pred[idx]    return x, x_pred, y_true, y_preddef rmse(y_true, y_pred):    return np.sqrt(((y_pred - y_true) ** 2).mean())def hausdorff(x_true, x_pred, y_true, y_pred):    return directed_hausdorff(np.c_[y_pred, x_pred], np.c_[y_true,x_true])[0]def draw_gt_vs_model(img, model, points_gt, keypoints=None, lineThickness = 3 , side=0, truncate = False):    # output image    new_im = np.copy(img).astype(np.uint8)    # colors for ground-truth and model    color_gt = (0, 0, 255)    color_pred = (0, 255, 0)    # get sorted coordinates    vmin = 0    vmax = img.shape[0]    if points_gt is not None:        x = points_gt[:, 0]        idx_sort = np.argsort(x)    if keypoints is not None:        vmin = np.min(keypoints[:,0])        vmax = np.max(keypoints[:, 0])    x_pred = np.asarray(np.arange(vmin, vmax), dtype=int)    y_pred = np.asarray(interpolate.splev(x_pred, model, der=0), dtype=int)    if truncate:        if side:            idx = y_pred < img.shape[1] // 2            x_pred = x_pred[idx]            y_pred = y_pred[idx]        else:            idx = y_pred >= img.shape[1] // 2            x_pred = x_pred[idx]            y_pred = y_pred[idx]    if points_gt is not None:        y_true = np.asarray(points_gt[idx_sort, 1], dtype=int)        # number of points in ground-truth        N = len(x)        # draw ground-truth points        for i in range(N):            cv2.circle(new_im, (y_true[i], x[i]), lineThickness//2, color_gt, -1)    # number of points in prediction    N = len(x_pred)    # draw predicted points    for i in range(1, N):        cv2.line(new_im, (y_pred[i - 1], x_pred[i] - 1), (y_pred[i], x_pred[i]), color_pred, lineThickness)    # if keypoints is not None, draw keypoints    if keypoints is not None:        keypoints = np.asarray(keypoints, dtype=int)        for i in range(len(keypoints)):            y = int(keypoints[i, 0])            x = int(keypoints[i, 1])            cv2.circle(new_im, (x, y), lineThickness*2, (255, 0, 0), -1)    # draw line in the middle of the image    middle = img.shape[1] // 2    cv2.line(new_im, (middle, 0), (middle, img.shape[0] - 1), (255,255,0), lineThickness//2)    return new_imdef draw_mta_model(img, model, keypoints=None, lineThickness = 2, vmin=None, vmax=None, truncate=True):    # output image    new_im = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)    x = keypoints[:, 0] if keypoints is not None else None    # colors for ground-truth and model    # color_gt = (0, 0, 255)    color_pred = (255, 255, 255)    if vmin is None:        vmin = x.min()    if vmax is None:        vmax = x.max()    x_pred = np.asarray(np.arange(vmin, vmax + 1), dtype=int)    y_pred = np.asarray(interpolate.splev(x_pred, model, der=0), dtype=int)    N = len(x_pred)    # draw predicted points    for i in range(1, N):        cv2.line(new_im, (y_pred[i - 1], x_pred[i] - 1), (y_pred[i], x_pred[i]), color_pred, lineThickness)    if keypoints is not None and truncate:        side = get_side(keypoints, img.shape[1])        if side:            new_im[:, (new_im.shape[1] // 2):] = 0        else:            new_im[:, :new_im.shape[1] // 2] = 0    new_im = keep_largest_component(new_im/255.)*255    return new_imdef spline_modeling(img_color, roi, n_keypoints, spline_order, pixel_error=15., return_steps=False, th_img=None, draw_models=False):    img = img_color[:,:,1]    assert(len(img.shape)==2)# img must be a two-dimensional array    # Enhance image using CLAHE    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))    enh_img = clahe.apply(np.asarray(img*255, dtype=np.uint8))/255.    # Segment    if th_img is None:        th_img = im_seg(img_color, roi)    # Get distance transform    dt_im = normalized_distance_transform(th_img)    # Add inverse intensities    idx = dt_im>0    intensities = 1-enh_img[idx]    width = dt_im[idx]    intensities = (intensities-intensities.min())/intensities.max()    width = (width - width.min())/width.max()    dt_im[idx] = (width+intensities)/2.    # Get points for model fitting    points = np.argwhere(dt_im > 0)    points = order_points(points)    # Get points weights from the distance transform image    weights = dt_im[points[:, 0], points[:, 1]]    # Get model    if draw_models:        model, inliers, datapoints, knots, draw_im = RANSAC_estimate_splines(points, n_keypoints, spline_order,                                                            epsilon=pixel_error, weights=weights, flag_print=0,                                                            th_img=th_img, draw_models=draw_models)    else:        model, inliers, datapoints, knots = RANSAC_estimate_splines(points, n_keypoints, spline_order,                                                        epsilon=pixel_error, weights=weights, flag_print=0, th_img=th_img)    if return_steps:        if draw_models:            return enh_img, th_img, dt_im, model, points[inliers], datapoints, knots, draw_im        return enh_img,th_img, dt_im, model, points[inliers], datapoints, knots    return model, points[inliers], datapoints, knotsdef closest_node(node, nodes):    distances = cdist([node], nodes)    idx_min = distances.argmin()    return nodes[idx_min]def closest_distance(node, nodes):    distances = cdist([node], nodes)    return distances.min()def mean_closest_distance(x_true, x_pred, y_true, y_pred):    points_true = np.c_[y_true, x_true]    points_pred = np.c_[y_pred, x_pred]    lst_distances = [closest_distance(points_pred[i], points_true) for i in range(len(points_pred))]    return np.mean(lst_distances)