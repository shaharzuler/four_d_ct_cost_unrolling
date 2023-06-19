# import numpy as np
# from scipy.ndimage import zoom
# from path import Path
# from torch.utils.data import Dataset
# import torch.nn.functional as F
# import scipy
# import torchio as tio



# class CardioCTDataset(Dataset):
#     def __init__(self, args, valid, root, sp_file, data_seg='trainval', do_aug=True, load_kpts=True, load_masks=False):
#         self.root = Path(root) # flow_data/task_02
#         self.sp_file = Path(sp_file) # flow_data/task_02test_pairs_val_csv
#         self.data_seg = data_seg # valid
#         self.csv_loader = CSVLoader()
#         self.nifti_loader = NiftiLoader()
#         self.l2r_kpts_loader = L2RLmsLoader()
#         self.load_kpts = load_kpts
#         self.load_masks = load_masks
#         self.valid = valid
#         self.augment = DataAugmentor(do_aug, 
#                                     in_shape = args.orig_shape, 
#                                     out_shape = args.aug_shape if not valid else args.test_shape, 
#                                     valid=valid)
#         # self.samples = self.collect_samples()

#     def collect_samples(self):
#         samples = []
#         scans_dir = self.root / 'training' / 'scans'
#         ktps_dir = self.root / 'keypoints'
#         scans_list = scans_dir.files('*.gz')
#         scans_list.sort()
#         pairs = self.load_valid_pairs(csv_file=self.sp_file)

#         for idx in range(0,len(scans_list),2):
#             file_name = scans_list[idx].parts()[-1]
#             csid = int(file_name.split('_')[1])
#             if self.data_seg != 'trainval':
#                 if self.data_seg == 'train' and csid in pairs['fixed']:
#                     continue
#                 if self.data_seg == 'valid' and csid not in pairs['fixed']:
#                     continue
#             sc_pair = [scans_list[idx], scans_list[idx+1]]
#             sample = {'imgs': sc_pair}
#             sample['case'] = 'case_{:03d}'.format(csid)
#             try:
#                 assert all([p.isfile() for p in sample['imgs']])
                
#                 if self.load_masks: # False
#                     mask_dirs = [Path(sc.replace('scans','lungMasks')) for sc in sc_pair]
#                     sample['masks'] = mask_dirs
#                     assert all([p.isfile() for p in sample['masks']])

#                 if self.load_kpts: # False
#                     sample['kpts'] = ktps_dir / 'case_{:03d}.csv'.format(csid)
#                     assert sample['kpts'].isfile()

#             except AssertionError:
#                 print('Incomplete sample for: {}'.format(sample['imgs'][0]))
#                 continue

#             samples.append(sample)
#         return samples
    
#     def load_valid_pairs(self,csv_file):
#         pairs = self.csv_loader.load(fname=csv_file)
#         return {k: [dic[k] for dic in pairs] for k in pairs[0]}
    
#     def _load_sample(self, s):
#         images  = [self.nifti_loader.load_image(p) for p in s['imgs']]

#         target = {'case' : s['case']}
#         if 'kpts' in s:
#             target['kpts'] = self.l2r_kpts_loader.load(fname=s['kpts'])
#         if 'masks' in s:
#             masks = [self.nifti_loader.load_image(m) for m in s['masks']]
#             target['masks'] = masks

#         return images, target, 

#     def __len__(self):
#         return 1 ###### FIXME
#         # return len(self.samples)

#     def __getitem__(self, idx):
#         # images, target = self._load_sample(self.samples[idx])
#         # images, target = self.augment(images, target)

#         fourD_scans_dir_name1 = r"/home/shahar/projects/4dct_data/20/20/Anonymized - 859733/Ctacoc/DS_CorCTABi 1.5 B25f 0-95% Matrix 256 - 12" # r"/home/shahar/projects/4dct_data/44/Anonymized - 4669044/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 8"#  # r"/home/shahar/projects/4dct_data/44/Anonymized - 4669044/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 8"
#         fourD_ind1 = 10 # min(1*idx+7,17)#27) ###
#         fourD_frame1 = self.get_4D_frame(fourD_scans_dir_name1, fourD_ind1)

#         fourD_scans_dir_name2 = r"/home/shahar/projects/4dct_data/20/20/Anonymized - 859733/Ctacoc/DS_CorCTABi 1.5 B25f 0-95% Matrix 256 - 12" #r"/home/shahar/projects/4dct_data/44/Anonymized - 4669044/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 8" #
#         fourD_ind2 = 20 # min(1*idx+7,17)#27) ###
#         fourD_frame2 = self.get_4D_frame(fourD_scans_dir_name2, fourD_ind2)

#         fourD_frame1 = scipy.ndimage.zoom(fourD_frame1,zoom=(1,1,fourD_frame1.shape[0]/fourD_frame1.shape[2]),order=2)
#         fourD_frame2 = scipy.ndimage.zoom(fourD_frame2,zoom=(1,1,fourD_frame2.shape[0]/fourD_frame2.shape[2]),order=2)

#         voxel_size1 = [1.,1.,1.]
#         voxel_size2 = [1.,1.,1.]
#         images = [ [fourD_frame1, tuple(voxel_size1)] , [fourD_frame2, tuple(voxel_size2) ] ] #(1.75, 1.25, 1.75)

#         target = {'case':"case_0"}
#         images, target = self.augment(images, target)

#         data = {'imgs' : images, 
#                 'target' : target}

#         return data 
    
#     def augment_2nd_image(self, ct_img, gap=10, axis=0):
#         ct_img_2 = np.concatenate([ct_img[:,:,gap:].copy(),np.zeros_like(ct_img[:,:,:gap])],axis=axis)
#         return ct_img_2

#     def rescale_intensity(self, img3d):
#         img3d = np.expand_dims(img3d,0)
#         img3d = tio.RescaleIntensity(out_min_max=(-1, 1))(img3d)
#         img3d = tio.RescaleIntensity(out_min_max=(-0.98, 1), masking_method=lambda x: x >-0.99)(img3d) 
#         img3d = tio.RescaleIntensity(out_min_max=(-0.98, 1), masking_method=lambda x: x<0.15)(img3d)  #######
#         return img3d

#     def rotate_scan(self,img3d):
#         img3d = np.rot90(img3d,k=3)
#         img3d = np.rot90(img3d,k=2,axes=(1,2)) 
#         img3d = np.rot90(img3d,k=2,axes=(0,1)) 
#         return img3d
