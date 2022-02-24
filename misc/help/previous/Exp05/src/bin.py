#
# def get_first_of_dicom_field_as_int(x):
#     if type(x) == pydicom.multival.MultiValue:
#         return int(x[0])
#     else:
#         return int(x)
#
#
# def get_windowing(data):
#     dicom_fields = [data.WindowCenter, data.WindowWidth, data.RescaleSlope, data.RescaleIntercept]
#     return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
#
#
# def window_image(img, window_center, window_width, slope, intercept):
#     img = (img * slope + intercept)
#     img_min = window_center - window_width // 2
#     img_max = window_center + window_width // 2
#     img[img < img_min] = img_min
#     img[img > img_max] = img_max
#     return img
#
#
# def to_multi_ch(input_path):
#     dcm = pydicom.dcmread(input_path)
#     p1 = get_windowing(dcm)
#     i1 = normalize(window_image(dcm.pixel_array, *p1))
#     p2 = [p1[0]*.5, p1[1]*.5, p1[2], p1[3]]
#     i2 = normalize(window_image(dcm.pixel_array, *p2))
#     p3 = [p1[0]*1.5, p1[1]*1, p1[2], p1[3]]
#     i3 = normalize(window_image(dcm.pixel_array, *p3))
#     return np.stack([i1, i2, i3], axis=-1)
#
#
# def crop(x):
#     if x.ndim == 2:
#         return x[700:, 500:2500]
#     else:
#         return x[700:, 500:2500, :]
#
#
#
# def print_shape():
#     import os
#     from glob import glob
#     import pydicom
#     TRAIN_DIR = "/data/train"
#     dirs = os.listdir(TRAIN_DIR)
#
#     def get_first_of_dicom_field_as_int(x):
#         if type(x) == pydicom.multival.MultiValue:
#             return int(x[0])
#         else:
#             return int(x)
#
#     def get_windowing(data):
#         dicom_fields = [data.WindowCenter, data.WindowWidth, data.RescaleSlope, data.RescaleIntercept]
#         return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
#
#     # test_files = glob(join(TRAIN_DIR , "*"))
#     test_files = [join(TRAIN_DIR, d, d+".dcm") for d in dirs]
#
#     for i, f in enumerate(test_files):
#         _f = pydicom.dcmread(f)
#         try:
#             lst = get_windowing(_f)
#             print(lst)
#         except:
#             print(None)
#
#
#
# def print_shape():
#     import os
#     from glob import glob
#     import pydicom
#     TEST_DIR = "/data/test"
#
#     def get_first_of_dicom_field_as_int(x):
#         if type(x) == pydicom.multival.MultiValue:
#             return int(x[0])
#         else:
#             return int(x)
#
#     def get_windowing(data):
#         dicom_fields = [data.WindowCenter, data.WindowWidth, data.RescaleSlope, data.RescaleIntercept]
#         return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
#
#     # test_files = glob(join(TRAIN_DIR , "*"))
#     test_files = glob(join(TEST_DIR, "*"))
#
#     for i, f in enumerate(test_files):
#         _f = pydicom.dcmread(f)
#         try:
#             lst = get_windowing(_f)
#             print(lst)
#         except:
#             print(None)
#
#
# class DataLoader(Sequence):
#     def __init__(self,
#                  mode,
#                  data_dir,
#                  valid_ratio=.1,
#                  batch_size=1,
#                  n_patches=32,
#                  patch_size=512,
#                  augmentation=None,
#                  **kwargs):
#         super(DataLoader, self).__init__()
#         assert mode in ["train", "valid"]
#
#         data_dirs = glob(join(data_dir, "*"))
#         input_paths = [glob(join(f, "*.dcm"))[0] for f in data_dirs]
#         target_paths = [sorted(glob(join(f, "*.png"))) for f in data_dirs]
#
#         train_X_paths, valid_X_paths, train_y_paths, valid_y_paths = train_test_split(
#             input_paths, target_paths, test_size=valid_ratio, random_state=2020
#         )
#
#         self.mode = mode
#         if mode == "train":
#             self.input_paths = train_X_paths
#             self.target_paths = train_y_paths
#         else:
#             self.input_paths = valid_X_paths
#             self.target_paths = valid_y_paths
#
#         self.indexes = np.arange(len(self.input_paths))
#         self.batch_size = batch_size
#         self.n_patches = n_patches
#         self.patch_size = patch_size
#         self.augmentation = augmentation
#         self.on_epoch_end()
#
#     def on_epoch_end(self):
#         np.random.shuffle(self.indexes)
#
#     def __len__(self):
#         return math.ceil(len(self.indexes) / self.batch_size)
#
#     def __getitem__(self, idx):
#         indexes = self.indexes[idx*self.batch_size : (idx+1)*self.batch_size]
#         if self.mode == "train":
#             batch_x, batch_y = self.train_process(indexes)
#         else:
#             batch_x, batch_y = self.valid_process(indexes)
#         return batch_x, batch_y
#
#     def train_process(self, indexes):
#         try:
#             inputs = to_multi_ch(self.input_paths[indexes[0]])
#         except:
#             inputs = pydicom.dcmread(self.input_paths[indexes[0]]).pixel_array
#             inputs = normalize(np.stack([inputs, inputs, inputs], axis=-1))
#         inputs = inputs.astype(np.float32)
#         targets = [np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in self.target_paths[indexes[0]]]
#         targets = normalize(np.stack(targets, axis=-1)).astype(np.float32)
#
#         common_size = [self.batch_size*self.n_patches, self.patch_size, self.patch_size]
#         batch_x = np.zeros(common_size + [3], dtype=np.float32)
#         batch_y = np.zeros(common_size + [8], dtype=np.float32)
#
#         for i in range(self.n_patches):
#             while True:
#                 patch_x, patch_y = random_crop(inputs, targets, self.patch_size)
#                 if not np.all(patch_y == 0):
#                     break
#
#             if self.augmentation is not None:
#                 aug_ = self.augmentation(image=patch_x, mask=patch_y)
#                 batch_x[i] = aug_["image"]
#                 batch_y[i] = aug_["mask"]
#             else:
#                 batch_x[i] = patch_x
#                 batch_y[i] = patch_y
#
#         return batch_x, batch_y
#
#     # 고정 셋으로 수정 필요
#     def valid_process(self, indexes):
#         try:
#             inputs = to_multi_ch(self.input_paths[indexes[0]])
#         except:
#             inputs = pydicom.dcmread(self.input_paths[indexes[0]]).pixel_array
#             inputs = normalize(np.stack([inputs, inputs, inputs], axis=-1))
#         inputs = inputs.astype(np.float32)
#         targets = [np.array(cv2.imread(i, cv2.IMREAD_GRAYSCALE)) for i in self.target_paths[indexes[0]]]
#         targets = normalize(np.stack(targets, axis=-1)).astype(np.float32)
#
#         common_size = [self.batch_size*self.n_patches, self.patch_size, self.patch_size]
#         batch_x = np.zeros(common_size + [3], dtype=np.float32)
#         batch_y = np.zeros(common_size + [8], dtype=np.float32)
#
#         for i in range(self.n_patches):
#             while True:
#                 patch_x, patch_y = random_crop(inputs, targets, self.patch_size)
#                 if not np.all(patch_y == 0):
#                     break
#
#             batch_x[i] = patch_x
#             batch_y[i] = patch_y
#
#         return batch_x, batch_y
#
